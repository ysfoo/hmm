import numpy as np
import scipy.stats as stats
from helper import gen_prob_vecs, lse_cols


"""Discrete observations."""

class DiscreteEmission():
    def __init__(self, n_state, obs_size):
        self.n_state = n_state
        self.obs_size = obs_size
        self.obs_list = list(range(obs_size))
        self.params = None   
        self.rvs = None
        
        
    def gen_params(self, alpha_min=1, alpha_max=1):
        # params are log probs
        self.params = np.log(gen_prob_vecs(self.n_state, self.obs_size, alpha_min, alpha_max))
        
    
    def emit(self, state):
        return np.random.choice(self.obs_list, p=np.exp(self.params[state]))
    
    
    def logl(self, state, obs):
        return self.params[state,obs]
    
    
    def logl_vec(self, obs):
        return self.params[:,obs]
    
    
    def mstep(self, obs_seq, z_logps):
        # z_logps is seq_len * n_state
        obs_seq = np.array(obs_seq)
        
        dtor = lse_cols(z_logps)
        ntor = np.full((self.n_state, self.obs_size), -np.inf)
        for x in range(self.obs_size):
            sub_z = z_logps[obs_seq==x,:]
            if sub_z.size:
                ntor[:,x] = lse_cols(sub_z)
        
        self.params = ntor - dtor[:,np.newaxis]
        
        # return log_likelihood
        return np.exp(z_logps.ravel()).dot(self.params[:,obs_seq].ravel('F'))
    
    
"""Normally distributed observations."""

class GaussianEmission():
    def __init__(self, n_state, min_mean=-2, max_mean=2, min_var=0.25, max_var=1):
        self.n_state = n_state
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.min_var = min_var
        self.max_var = max_var
        self.params = None       
        
        
    def gen_params(self):
        # params are mean and variance
        minlogv = np.log(self.min_var)
        maxlogv = np.log(self.max_var)
        self.params = np.array([
                          [stats.uniform(self.min_mean, self.max_mean - self.min_mean).rvs(),
                           np.exp(stats.uniform(minlogv, maxlogv - minlogv).rvs())] for _ in range(self.n_state)])
    
    
    def emit(self, state):
        mean, var = self.params[state]
        return stats.norm(mean, np.sqrt(var)).rvs()
    
    
    def logl(self, state, obs):
        mean, var = self.params[state]
        return -0.5 * (np.log(2 * np.pi * var) + np.square(obs - mean) / var)
    
    
    def logl_vec(self, obs):
        return -0.5 * (np.log(2 * np.pi * self.params[:,1]) + np.square(obs - self.params[:,0]) / self.params[:,1])
    
    
    def mstep(self, obs_seq, z_logps):
        # z_logps is seq_len * n_state
        seq_len = len(obs_seq)
        obs_seq = np.array(obs_seq)
        
        dtor = lse_cols(z_logps)
        minval = obs_seq.min() - 1
        self.params[:,0] = np.exp(lse_cols(z_logps + np.log(obs_seq - minval)[:,np.newaxis]) - dtor) + minval
        self.params[:,1] = np.exp(lse_cols(z_logps + np.log(np.square(obs_seq[:,np.newaxis] - np.tile(self.params[:,0], (seq_len, 1))))) - dtor)
        
        return -0.5 * (seq_len * (np.log(2 * np.pi) + 1) + np.exp(dtor).dot(np.log(self.params[:,1])))     
        