
import numpy as np
import matplotlib
import math


class gauss_est:
   
    
    
    def __init__(self, h):
        '''Этот модуль позволяет находить фазу по значениям интенсивности.
     
     Данные должны буть представлены в следующем виде: X и Y это списки, длинны равной количеству каналов. Например, если у нас три канала, то X = [X_1, X_2, X_3] и Y = [Y_1, Y_2, Y_3]. 
     X_i и Y_i это набор данных на i-ом канале, тип этих массивов numpy. X_i[j] - это j-ая серия измерений на i-ом канале. 
     
     В predict данные посылаются следующим образом: Y - список измерений на каждом канале. Например, Y = [Y_1, Y_2, Y_3]. Y_i - это масиив типа numpy с измерениями амплитуды в канале. Возвращает функция три значения ans, proba, probability. ans - предсказанное значение фазы, proba - логарифм вероятности этого результата, probabilyty - массив numpy значений логартфма вероятности того, что измерено phis[i].
     
     '''
        
        
        
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
            