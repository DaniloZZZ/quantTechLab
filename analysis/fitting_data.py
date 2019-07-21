
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
        self.max = 0;
        self.min = 0;
        
        
        
        
    
    def get_koef(self, phi, X, Y):
        sigma = 0
        mu = 0
        n = 0
        mu = Y[abs(X-phi)<self.h].mean()
        sigma = Y[abs(X-phi)<self.h].std()
        return mu, sigma
    
    
    def learning_proces(self, X, Y):
        self.phis = np.linspace(self.min, self.max, math.floor((self.max-self.min)/self.h))
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
    
    def make_prediction(self, Y, probability = False):
        ans = self.min
        
        prob = self.count_metrics(np.array([self.mu[i][0] for i in range(len(self.mu))]), np.array([self.sigma[i][0] for i in range(len(self.sigma))]), Y)
        
        if probability:
            probab_number = np.zeros(len(self.phis))
        
        else: probab_number = 0
        
        
        for j in range(len(self.phis)):
            a = self.count_metrics(np.array([self.mu[i][j] for i in range(len(self.mu))]), np.array([self.sigma[i][j] for i in range(len(self.sigma))]), Y)
            if probability:
                probab_number[j] = a
            if a > prob :
                ans = self.phis[j]
                prob = a
        '''for i in range(len(phis)):
            probability[i] = self.count_metrics(mu, sigma, Y)'''
        return ans, prob, probab_number
    
    
    
    def min_max(self, lst):
        e = np.array(lst)
        return e.max(), e.min()
    
    def fit(self, X, Y):
        self.mu = []
        self.sigma = []
        self.max, self.min = self.min_max(X)
        for j in range(len(X)):
            mu_1, sigma_1 = self.learning_proces(X[j], Y[j])
            self.mu.append(mu_1)
            self.sigma.append(sigma_1)
       
            
            
    def predict(self, Y, probability = False):
        prob_result = []
        if type(Y) == np.ndarray:
            if type(Y[0]) == np.ndarray:
                X_predicted = np.zeros(len(Y[0]))
                proba_predicted = np.zeros(len(Y[0]))
                for i in range(len(Y[0])):
                    ans, prob, probab_number = self.make_prediction([Y[k][i] for k in range(len(Y))], probability = probability)
                    if probability:
                        prob_result.append(probab_number)
                    X_predicted[i] = ans
                    proba_predicted[i] = prob 
                return X_predicted, proba_predicted, np.array(prob_result)
            else :
                X_predicted = np.zeros(len(Y))
                proba_predicted = np.zeros(len(Y))
                for i in range(len(Y)):
                    ans, prob, probab_number = self.make_prediction([Y[i]])
                    X_predicted[i] = ans
                    proba_predicted[i] = prob 
                    if probability:
                        prob_result.append(probab_number)
                return X_predicted, proba_predicted, np.array(prob_result)
        else :    
            return self.make_prediction(Y)
    
    
    
    
    
    
class preprocessing:
    def __init__(self):
        self.a = 0
        
    def condition( a):
        return True
    
    
    def valid_data(self, X, Y, cond = condition, monotone = False):
        size = int(X.shape[0])
        X.flatten()
        Y.flatten()
        Y = Y[ cond(X)]
        X = X[cond(X)]
        ind = len(Y) % size
        if ind != 0:
            Y = Y[:-ind]
            X = X[:-ind]
        Y = Y.reshape(size,-1)
        X = X.reshape(size,-1)
        
        if monotone:
            X = list(X)
            Y = list(Y)
            for i in range(len(X)):
                j = 0
                while j < len(X[i])-2:
                    if X[i][j] < X[i][j+1]:
                        j += 1
                    else:
                        for k in range(1):
                            X[i] = np.delete(X[i], j)
                            Y[i] = np.delete(Y[i], j)
            X = np.array(X)
            Y = np.array(Y)
        return X, Y
    
    
    def train_test_split(self, X, Y, test_size = 0.2):
        DATA = np.array([X, Y])
        all_size = DATA.shape[2]
        idxs = np.arange(0, all_size)
        labl_idx = np.ones(all_size )*np.arange(0,all_size)/all_size
        test_data= DATA[:,:,labl_idx<test_size]
        train_data= DATA[:,:,labl_idx>test_size]
        X_train,Y_train = train_data
        X_test, Y_test= test_data
        return X_train, X_test, Y_train, Y_test
        
    def normalized(self, X, amplitude = 1):
        maximum = np.array([[X[i].max() for k in range(len(X[i]))] for i in range(len(X))])
        minimum = np.array([[X[i].min() for k in range(len(X[i]))] for i in range(len(X))])
        return (X - minimum)/(maximum-minimum)*amplitude
        
        
            