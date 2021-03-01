#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:11:42 2021

@author: kiran
"""


from numpy.random import randint
from quimb.tensor.optimize import TNOptimizer
from tqdm import tqdm


class SPSAWraper(TNOptimizer):
    """
    A wrapper for TNOPtimizer that uses all of it's nice stuff to handle params, 
    but runs spsa (not included in scipy) instead, see 
    quimb.tensor.optimize.TNOptimizer for full definitions"""
    
    def __init__(self, **args):
        super().__init__(**args)
        if self._method == 'SPSA':
            self.optimize = self._make_spsa_optimizer()
    
    
    def _make_spsa_optimizer(self):
        """
        Creates the method to overwrite the TNOptimizer.optimize method

        Returns
        -------
        A callable (spsa) optimize function based on qiskits implementation

        """
        def optimize(maxiter: int = 1000,
                     tol = None,
                     save_steps: int = 1,
                     c0: float = 0.62,
                     c1: float = 0.1,
                     c2: float = 0.602,
                     c3: float = 0.101,
                     c4: float = 0):
            """
            This method is heavily based on qiskits optimizers.spsa method, 
            adapted here to worth with on quibs tn's without exact gradients 

            Parameters
            ----------
            maxiter: Maximum number of iterations to perform.
            tol : None or float stops optim if tol is reached (default none - completes all steps)
            save_steps: Save intermediate info every save_steps step. It has a min. value of 1.
            last_avg: Averaged parameters over the last_avg iterations.
                If last_avg = 1, only the last iteration is considered. It has a min. value of 1.
            c0: The initial a. Step size to update parameters.
            c1: The initial c. The step size used to approximate gradient.
            c2: The alpha in the paper, and it is used to adjust a (c0) at each iteration.
            c3: The gamma in the paper, and it is used to adjust c (c1) at each iteration.
            c4: The parameter used to control a as well.
            
            Returns
            -------
            TYPE : updated object? (same return as TNOptimize)
            """
            _spsa_vars = [c0, c1, c2, c3, c4]
            theta = self.vectorizer.vector
            nb_params = len(theta)
            use_exact_grads = 'grads' in self._method
            
            if save_steps:
                theta_vec = [theta]
                cost_vec = [self.vectorized_value_and_grad(theta)[0]]
            
            
            pbar = tqdm(total=maxiter, disable=not self.progbar)
            def callback(_):
                pbar.clear()
                pbar.update()
                val = round(self.loss, 5)
                pbar.set_description(str(val))

                if self.loss_target is not None:
                    if self.loss < self.loss_target:
                        # returning True doesn't terminate optimization
                        raise KeyboardInterrupt
            
            for ii in range(maxiter):
            
                a_spsa = float(_spsa_vars[0]) / ((ii + 1 + _spsa_vars[4])**_spsa_vars[2])
                c_spsa = float(_spsa_vars[1]) / ((ii + 1)**_spsa_vars[3])
                delta = 2 * randint(0, 2, size=nb_params) - 1
                # plus and minus directions
                
                if use_exact_grads:
                    raise NotImplementedError('Will use grad calc to project on to SP-direction')
                else:
                    theta_plus = theta + c_spsa * delta
                    theta_minus = theta - c_spsa * delta

                    cost_plus = self.vectorized_value_and_grad(theta_plus)[0]
                    cost_minus = self.vectorized_value_and_grad(theta_minus)[0]
                    # derivative estimate
                    g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
                    # updated theta
                    theta = theta - a_spsa * g_spsa
                
                callback(ii)
                
                if tol is not None:
                    if (cost_plus + cost_minus)/2 < tol:
                        break
                    
                if save_steps:
                    theta_vec.append(theta)
                    cost_vec.append(cost_plus/2+cost_minus/2)
                
            
            result_dict = {'hyper_parameters':_spsa_vars,
                             'maxiter':maxiter,
                             'theta_opt':theta,
                             'cost_opt':self.vectorized_value_and_grad(theta)[0],
                             'grad_opt':self.vectorized_value_and_grad(theta)[1]}
            if save_steps:
                result_dict['theta_history'] = theta_vec
                result_dict['cost_history'] = cost_vec
            self.result_dict = result_dict
            pbar.close()

            return self.inject_res_vector_and_return_tn()
        return optimize





#%% quick testing to see it works
if __name__ == "__main__":
    import quimb as qu
    import quimb.tensor as qtn
    
    # construct 10 qubit heis model and example MPS state
    L = 5
    H = qu.ham_heis(L, sparse=True, cyclic=False)
    gs = qu.groundstate(H)
    target = qtn.Dense1D(gs)
    mps = qtn.MPS_rand_state(L, 2, cyclic=False)
    
    
    # define norm and infidelity function
    def normalize_state(psi):
        return psi / (psi.H @ psi) ** 0.5
    
    def negative_overlap(psi, target):
        return 1 - abs(psi.H @ target) ** 2  # minus so as to minimiz


    # create optimizer just like TNOptimizer but supply 'SPSA' method
    optimizer = SPSAWraper(tn=mps,
                           loss_fn=negative_overlap,
                           norm_fn=normalize_state,
                           loss_constants={'target': target},
                           autodiff_backend='tensorflow', 
                           optimizer='SPSA')
    #run 50 iterations of spsa
    optimizer.optimize(100, tol= 1e-1)
    results = optimizer.result_dict
    print('final result is {}'.format(results['cost_opt']))



