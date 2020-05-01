import numpy as np
from helper import gen_prob_vecs, lse, lse_rows, lse_cols


"""Wrapper for HMM model."""

class HMM:
    def __init__(self, n_state, emission_model):
        self.n_state = n_state
        self.states = list(range(n_state))
        self.emission_model = emission_model
        
        self.init_prob = None
        self.transition_mat = None
        self.log_pi = None
        self.log_amat = None
        
        self.state_seq = None
        self.obs_seq = None
        
        
    def gen_params(self, alpha_min=1, alpha_max=1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        self.init_prob = [1 / self.n_state] * self.n_state
        self.transition_mat = gen_prob_vecs(self.n_state, self.n_state, alpha_min, alpha_max)
        self.log_pi = np.log(self.init_prob)
        self.log_amat = np.log(self.transition_mat)
        self.emission_model.gen_params()
        
        
    def random_seq(self, seq_len, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        self.state_seq = self.gen_state_seq(seq_len)
        self.obs_seq = [self.emission_model.emit(state) for state in self.state_seq]
        
        return self.obs_seq
        
        
    def gen_state_seq(self, seq_len):  
        prev = np.random.choice(self.states, p=self.init_prob)
        state_seq = [prev]

        for _ in range(1, seq_len):
            prev = np.random.choice(self.states, p=self.transition_mat[prev])
            state_seq.append(prev)
            
        return state_seq
    
    
    def viterbi(self, obs_seq):
        seq_len = len(obs_seq)        
        max_vals = self.log_pi + self.emission_model.logl_vec(obs_seq[0])
        max_args = []
        
        for t in range(1, seq_len):
            logls = self.log_amat + max_vals[:,np.newaxis] + self.emission_model.logl_vec(obs_seq[t])
            max_vals = logls.max(axis=0)
            max_args.append(logls.argmax(axis=0))
            
        state_seq = [max_vals.argmax()]
        for args in reversed(max_args):
            state_seq.append(args[state_seq[-1]])
            
        return state_seq[::-1] 
        
    
    def baum_welch(self, obs_seq, n_iter):
        print('iterations:', end=' ')
        first = True
        iter_params = [self.emission_model.params.copy()]
        for n in range(n_iter):
            logl, loggms, logxis = self.e_step(obs_seq)
            self.m_step(obs_seq, loggms, logxis)
            iter_params.append(self.emission_model.params.copy())
            if first:
                first = False
            else:
                if logl < prev:
                    if abs(logl - prev) < 1e-8:
                        print(n + 1)
                        return logl, iter_params
                    assert logl >= prev
                    
            if n % 50 == 50 - 1:
                print(n + 1, end=' ')
            #print(f'{logl:.4f}')
            #print(logl)
            #print(self.init_prob)
            #print(self.transition_mat)
            #print(np.exp(self.emission_model.params))
            #print(self.emission_model.params)
            
            prev = logl
        print()
            
        return logl, iter_params
            
            
    def e_step(self, obs_seq):
        seq_len = len(obs_seq)
        emission_logl_mat = np.array([self.emission_model.logl_vec(obs) for obs in obs_seq])
        
        logas = self.get_alphas(obs_seq, emission_logl_mat)
        logbs = self.get_betas(obs_seq, emission_logl_mat)
        
        # log gamma
        logabs = logas + logbs
        loggms = logabs - lse_rows(logabs)[:,np.newaxis]
        
        # log xi
        logxis = np.zeros((seq_len - 1, self.n_state, self.n_state))        
        for t in range(1, seq_len):   
            ntor = self.log_amat + logas[t - 1][:,np.newaxis] + logbs[t] + emission_logl_mat[t]
            logxis[t - 1] = ntor - lse(ntor.ravel())
            
        return lse(logas[seq_len - 1]), loggms, logxis    
    
    
    def m_step(self, obs_seq, loggms, logxis):
        self.log_pi = loggms[0] 
        self.log_amat = lse_cols(logxis) - lse_cols(loggms[:-1,:])[:,np.newaxis]
        
        # update exponentiated equivalents
        self.init_prob = np.exp(self.log_pi)
        self.transition_mat = np.exp(self.log_amat) 
        
        # check if probs add to 1
        #print(lse(self.log_pi))
        #print(lse_rows(self.log_amat))
        #print(lse_rows(self.emission_model.params))
        
        emission_logl = self.emission_model.mstep(obs_seq, loggms)
        
        # calculate log likelihood        
        # return np.exp(loggms[0]).dot(self.log_pi) + np.exp(lse_cols(logxis)).ravel().dot(self.log_amat.ravel()) + emission_logl  
    
    
    def get_alphas(self, obs_seq, emission_logl_mat):
        seq_len = len(obs_seq)
        logas = np.zeros((seq_len, self.n_state))
        
        logas[0] = self.log_pi + emission_logl_mat[0]
        for t in range(1, seq_len):
            logas[t] = lse_cols(self.log_amat + logas[t - 1][:,np.newaxis] + emission_logl_mat[t])
        
        return logas
    
    
    def get_betas(self, obs_seq, emission_logl_mat):
        seq_len = len(obs_seq)
        logbs = np.zeros((seq_len, self.n_state))
        
        for t in range(seq_len - 1, 0, -1):
            logbs[t - 1] = lse_rows(self.log_amat + logbs[t] + emission_logl_mat[t])
                   
        return logbs
        
       
    


        
        
        
    
        

