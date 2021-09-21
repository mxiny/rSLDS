from re import X
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import cov
from scipy.stats import invwishart
from pypolyagamma import PyPolyaGamma

import utils

class rSLDS:
    '''
        recurrent Switching Linear Dynamical System
    '''

    def __init__(self, K, N, M, T) -> None:
        '''
        Argument:
            - K (Integer): number of modes
            - N (Integer): dimension of observations
            - M (Integer): dimension of latent varibles
            - T (Integer): trial length
        
        Return:
            - None
        '''

        self.K = K
        self.N = N
        self.M = M
        self.T = T

        # Switching
        # Transition matrix initialized with Dirichlet distribution
        alpha = 1 / K * np.ones(K)
        self.Pis = np.stack([np.random.dirichlet(alpha) for k in range(K)])

        # Dynamics
        # Dynamical parameters initialized with Normal-inverse-Wishart distribution
        lambd = M
        scale = np.eye(M)
        self.As = np.stack([invwishart.rvs(df=lambd, scale=scale, size=(M, M)) for k in range(K)])
        self.bs = np.stack([invwishart.rvs(df=lambd, scale=scale, size=(M)) for k in range(K)])
        self.Qs = np.stack([invwishart.rvs(df=lambd, scale=scale, size=(M, M)) for k in range(K)])
        
        # Emission
        # All discrete states share the same C, d, S
        eta = N
        self.C = invwishart.rvs(df=eta, scale=scale, size=(N, N))
        self.d = invwishart.rvs(df=eta, scale=scale, size=(N))
        self.S = invwishart.rvs(df=eta, scale=scale, size=(N, N))

        # Stick breaking parameters
        self.Rs = np.random.random((K, K - 1, M))
        self.rs = np.random.random((K, K - 1))


    def _stick_breaking_logistic_regression(self, zt, xt):
        vt = np.dot(self.Rs[zt], xt) + self.rs[zt]

        Pit = np.ones(self.K)
        for k in range(self.K - 1):
            Pit[k] = utils.sigmoid(vt[k])
            for j in range(k):
                Pit[k] *= utils.sigmoid(-vt[j])
        
        for k in range(self.K - 1):
            Pit[-1] *= utils.sigmoid(-vt[k])
        return Pit
    
    def _transmition(self, zt, xt):
        Pit = self._stick_breaking_logistic_regression(zt, xt)
        z_next = np.random.choice(self.K, p=Pit)
        return z_next

    def _dynamic(self, zt, z_next, xt):
        vt = np.dot(self.Rs[zt], xt) + self.rs[zt]

        pg = PyPolyaGamma(seed=0)
        omegas = np.stack([pg.pgdraw(z_next >= k, vt[k]) for k in range(self.K - 1)])

        cov = 1 / np.diag(omegas)
        aux_k = np.stack([((z_next == k) - 0.5 * (z_next >= k)) for k in range(self.K - 1)])
        mean = np.dot(cov, aux_k)

        x_next = np.random.multivariate_normal(mean=mean, cov=cov)
        
        return x_next

    def _emission(self, xt):
        wt = np.random.multivariate_normal(mean=0, cov=self.S)
        yt = np.dot(self.C, xt) + self.d + wt

        return yt

                
    def fit(self, data, max_iter):

        return 0

    def infer(self, data):
        return 0

    def sample(self, new_len):
        z = np.zeros(new_len, dtype=int)
        x = np.zeros((new_len, self.M))
        y = np.zeros((new_len, self.N))
        z[0] = np.random.choice(self.K, p=self.Pis[0])
        x[0] = self._dynamic(z[0], z[0], x[0])

        for i in range(1, new_len):
            z[i] = self._transmition(z[i - 1], x[i - 1])
            x[i] = self._dynamic(z[i - 1], z[i], x[i - 1])
            y[i] = self._emission(x[i])
        
        return z, x, y
    
    def predict(new_len):

        return 0

