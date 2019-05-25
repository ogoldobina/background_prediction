import numpy as np
from skimage.io import imread
from scipy.special import logsumexp
from tqdm import tqdm, tqdm_notebook



class EM_algorithm:
    def __init__(self, diag=False, num_components=3, num_iter=20, num_tries=3):
        self.diag = diag
        self.num_components = num_components
        self.num_iter = num_iter
        self.num_tries = num_tries
        
    def fit(self, X):
        '''
        X - matrix N x D
        '''
        D = X.shape[1]
        max_ll = -np.inf

        for t in range(self.num_tries):
            w = np.random.randint(1,10,self.num_components)
            w = w / np.sum(w)
            mu = np.random.rand(self.num_components, D)
            if self.diag:
                sigma = np.ones((self.num_components, D))
            else:
                sigma = np.broadcast_to(np.eye(D), (self.num_components, D, D)).copy()

            centered_X = X[np.newaxis, :, :] - mu[:, np.newaxis, :] #KND

            LLs = []
            for i in range(self.num_iter):
                #E:
                if self.diag:
                    inv_sigma = 1 / sigma
                    log_P_X = (-0.5 * np.sum((centered_X ** 2) *
                            inv_sigma[:,np.newaxis,:], axis=2) - 
                            0.5 * (D * np.log(2 * np.pi) + 
                            np.log(np.prod(sigma, axis=1))[:, np.newaxis]))            
                else:
                    inv_sigma = np.linalg.inv(sigma) #KDD
                    log_P_X = (-0.5 * np.sum(np.matmul(centered_X, inv_sigma) * centered_X, axis=2) -
                            0.5 * (D * np.log(2 * np.pi) +
                            np.log(np.linalg.det(sigma)))[:, None])
                log_P_X_T = np.log(w)[:, np.newaxis] + log_P_X

                log_sum_exp = logsumexp(log_P_X_T, axis=0, keepdims=True)
                log_likelihood = log_sum_exp.sum()
                LLs += [log_likelihood]

                log_P_T = log_P_X_T - log_sum_exp
                P_T = np.exp(log_P_T)   ##KN

                #M:
                w = np.mean(P_T, axis=1)
                mu = np.dot(P_T, X) / np.sum(P_T, axis=1, keepdims=True)
                centered_X = X[np.newaxis, :, :] - mu[:, np.newaxis, :]
                if self.diag:
                    sigma = (np.sum((centered_X ** 2) * P_T[:,:,np.newaxis], axis=1) /
                    np.sum(P_T, axis=1)[:, np.newaxis] + 1e-3 * np.ones((self.num_components, D)))
                else:
                    for k in range(self.num_components):
                        sigma[k] = ((centered_X[k].T).dot(centered_X[k] * P_T[k][:, np.newaxis]) /
                                    P_T[k].sum() + 1e-3 * np.eye(D))
                        
            if log_likelihood > max_ll:
                max_ll = log_likelihood
                self.w = w
                self.mu = mu
                if self.diag:
                    sigma = np.apply_along_axis(np.diag, axis=1, arr=sigma)
                self.sigma = sigma
                self.log_likelihood = LLs
                self.P_T = P_T
                
    def predict_proba(self, X):
        '''
        X - matrix N x D
        '''
        centered_X = X[np.newaxis, :, :] - self.mu[:, np.newaxis, :]
        if self.diag:
            inv_sigma = 1 / self.sigma
            log_P_X = (-0.5 * np.sum((centered_X ** 2) *
                    inv_sigma[:,np.newaxis,:], axis=2) - 
                    0.5 * (X.shape[1] * np.log(2 * np.pi) + 
                    np.log(np.prod(self.sigma, axis=1))[:, np.newaxis]))            
        else:
            inv_sigma = np.linalg.inv(self.sigma) #KDD
            log_P_X = (-0.5 * np.sum(np.matmul(centered_X, inv_sigma) * centered_X, axis=2) -
                    0.5 * (X.shape[1] * np.log(2 * np.pi) +
                    np.log(np.linalg.det(self.sigma)))[:, None])
        log_P_X_T = np.log(self.w)[:, np.newaxis] + log_P_X
        log_sum_exp = logsumexp(log_P_X_T, axis=0)
        return log_sum_exp


class Background_Estimation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pixels_EM = None
        
    def fit(self, train_video):
        '''
        train_video - matrix T x N x D
        '''
        T, N, D = train_video.shape
        self.pixels_EM = np.empty((N, D), dtype=EM_algorithm)
        for n in tqdm_notebook(range(N), 'Background_Estimation.fit'):
            for d in range(D):
                em = EM_algorithm(**self.kwargs)
                em.fit(train_video[:, n, d][:, np.newaxis])
                self.pixels_EM[n,d] = em
                
    def predict(self, test_video, threshold=3):
        '''
        test_video, mask - matrix  T x N x D
        '''
        assert self.pixels_EM is not None, 'Not fitted yet'
        
        T, N, D = test_video.shape
        mask = np.empty((T, N, D))
        for n in range(N):
            for d in range(D):
                mu = self.pixels_EM[n,d].mu[0]
                sigma = self.pixels_EM[n,d].sigma[0][0]
                mask[:,n,d] = np.abs(test_video[:,n,d] - mu) >= threshold * np.sqrt(sigma)
        return mask
    
    def predict_fly(self, test_video, ro=0.1, threshold=3):
        assert self.pixels_EM is not None, 'Not fitted yet'
        
        T, N, D = test_video.shape
        mask = np.empty((T, N, D))
        
        for t in tqdm_notebook(range(T), 'predict_fly'):
            for n in range(N):
                for d in range(D):
                    mu = self.pixels_EM[n,d].mu[0]
                    sigma = self.pixels_EM[n,d].sigma[0][0]
                    res = np.abs(test_video[t,n,d] - mu) >= threshold * np.sqrt(sigma)
                    mask[t,n,d] = res
                    if not res:
                        bright = test_video[t,n,d]
                        new_nu = ro * bright + (1 - ro) * mu
                        self.pixels_EM[n,d].mu[0] = new_nu
                        new_sigma = (((bright - new_nu) ** 2) * ro +
                                                           (1 - ro) * sigma)
                        if new_sigma < 1e-3:
                            new_sigma = 1e-3
                        self.pixels_EM[n,d].sigma[0][0] = new_sigma
        return mask
        


class Background_Estimation_rgb:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pixels_EM = None
        
    def fit(self, train_video):
        '''
        train_video - matrix T x N x D x 3
        '''
        T, N, D, K = train_video.shape
        self.pixels_EM = np.empty((N, D), dtype=EM_algorithm)
        for n in tqdm_notebook(range(N), 'Background_Estimation.fit'):
            for d in range(D):
                em = EM_algorithm(**self.kwargs)
                em.fit(train_video[:, n, d, :])
                self.pixels_EM[n,d] = em
                
    def predict(self, test_video, threshold=3):
        '''
        test_video, mask - matrix  T x N x D x 3
        '''
        assert self.pixels_EM is not None, 'Not fitted yet'
        
        T, N, D, K = test_video.shape
        mask = np.empty((T, N, D))
        for n in range(N):
            for d in range(D):
                mask[:,n,d] = self.pixels_EM[n,d].predict_proba(test_video[:,n,d,:]) < threshold
        return mask