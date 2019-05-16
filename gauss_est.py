
import numpy as np
import matplotlib
import math


class gauss_est:
    def __init__(self, h):
        self.h = h
        self.mu = []
        self.sigma = []
    
    def get_koef(self, phi, X, Y):
        sigma = 0
        mu = 0
        n = 0
        mu = Y[abs(X-phi)<self.h].mean()
        sigma = Y[abs(X-phi)<self.h].std()
        return mu, sigma
    
    
    def learning_proces(self, X, Y):
        self.phis = np.linspace(0, 2*3.1415, math.floor(3.1415/self.h))
        mu = np.zeros(len(self.phis))
        sigma = np.zeros(len(self.phis))
        for i in range(len(self.phis)):
            mu[i] , sigma[i] = self.get_koef(self.phis[i], X, Y)
        return mu, sigma
    
    def count_metrics(self, mu, sigma, Y):
        r = 0
        for i in range(len(Y)):
            r -= ((Y[i]-mu[i])**2/2/sigma[i]).sum()
        return r
    
    def make_prediction(self, Y):
        ans = 0

        prob = self.count_metrics(np.array([self.mu[i][0] for i in range(len(self.mu))]), np.array([self.sigma[i][0] for i in range(len(self.sigma))]), Y)
        for j in range(len(self.phis)):
            a = self.count_metrics(np.array([self.mu[i][j] for i in range(len(self.mu))]), np.array([self.sigma[i][j] for i in range(len(self.sigma))]), Y)
            if a > prob :
                ans = self.phis[j]
                prob = a
        '''for i in range(len(phis)):
            probability[i] = self.count_metrics(mu, sigma, Y)'''
        return ans, prob
    
    
    def fit(self, X, Y):
        for j in range(len(X)):
            mu_1, sigma_1 = self.learning_proces(X[j], Y[j])
            self.mu.append(mu_1)
            self.sigma.append(sigma_1)
            
            
    def predict(self, Y):
        return self.make_prediction(Y)
            