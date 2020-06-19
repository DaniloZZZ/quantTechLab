
import numpy as np
import matplotlib
import math
import os
import imageio

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
#             r -= ((Y[i]-mu[i])**2/2).sum()
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
        e = np.array([item.max() for item in lst])
        i = np.array([item.min() for item in lst])
        return e.max(), i.min()
    
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
    
    
    
    
class data_from_dataset():
    def __init__(self, dataset):
        self.intensities = np.array([i.mean() for i in dataset.images])
        self.target = dataset.target
    def get(self):
        return self.target, self.intensities    
    
    
    
    
    
class dataset_from_folder():
    def __init__(self, paths):
        
        self.paths = paths
       
    def get(self):
        itensities = []
        phases = []
        for path in self.paths:
            images, target = self.get_image_target(path)
            intensity = np.array([item.mean() for item in images])
            itensities.append(intensity)
            phases.append(target)
        return np.array(phases), np.array(itensities)

    
    def get_items(self,target, images, idx):
        a = target[idx]
        b = 0
        cnt = 0
        while idx+cnt < len(target) and target[idx+cnt] == a :
            b += images[idx+cnt]
            cnt += 1
        b = b/cnt
        return a, b, idx+cnt
    
    def get_average_data(self, target, images):
        new_target = []
        new_images = []
        idx = 0
        while idx < len(target):
            a, b, idx = self.get_items(target, images, idx)
            new_target.append(a)
            new_images.append(b)
        return np.array(new_target), np.array(new_images)
    
    def get_image_target(self, path):
            names = os.listdir(path)
            need = lambda x: ".png" in x
            names = list(filter(need, names))
            names.sort(key= lambda x: int(x.split("_")[0]))
            images = []
            target = []
            for item in names:
                images.append(np.array(imageio.imread(str(path/item))))
                target.append(float(item.split("_")[1]))
            images = np.array(images)
            target = np.array(target)
            return images, target
            
            
    def split(self,start = 0, end = 1,shuffle = False, test_size = 0.2, filtered = None):
        self.names = []
        self.filtered = filtered
        for path in self.paths:
            self.names.append(self.get_names(path))
        
        self.names = np.array(self.names)
        train_names = []
        test_names = []
        
        for item in self.names:
            train, test = self.split_names(item, start = start, end = end, shuffle = shuffle, test_size = test_size)
            train_names.append(train)
            test_names.append(test)
        train_dataset = self.Generata_data(self.paths, train_names, filtered = self.filtered)
        test_dataset = self.Generata_data(self.paths, test_names, filtered = self.filtered, test = True)
        return train_dataset, test_dataset
        
        
    def get_names(self,path):
        names = os.listdir(path)
        
        need = lambda x: ".png" in x
        names = list(filter(need, names))
        
        names.sort(key= lambda x: int(x.split("_")[0]))
       
        return names
    def split_names(self, names, shuffle = False, test_size = 0.2, start = 0, end = 1):
        names = np.array(names)
        idx = np.linspace(start*(len(names)-1),end*(len(names)-1), len(names), dtype = int)
        if shuffle:
            random.shuffle(idx)
        
        return names[idx[:int((1-test_size)*len(idx))]], names[idx[:int((test_size)*len(idx))]]
    
    def Generata_data(self,paths, names, filtered = None, test = False):
        if test:
            new_names = []
            for item in names:
                k = 0
        
                for i in range(len(item)):
                    if item[i+1].split("_")[1] < item[i].split("_")[1]:
                        k = i
                        break

                item = item[k+1:]
                for i in range(len(item)-1, 0, -1):
                    if item[i-1].split("_")[1] > item[i].split("_")[1]:
                        k = i
                        break

                item = item[:k-1]
                new_names.append(item)
            names = np.array(new_names)
         
        target = []
        inten = []
        for path, name in zip(paths, names):
            target0, inten0 = self.get_inten_of_images(name, path, filtered = filtered)
            target.append(target0)
            inten.append(inten0)
            
        
        
        if test:
            i = 0
            for target0, inten0 in zip(target, inten):
                target[i], inten[i] = self.get_average_data(target0, inten0)
                i += 1
            
            
            new_target = self.make_final_target(target)
            new_inten = np.zeros((len(new_target), len(target)))
            idxs = np.zeros(len(target), dtype = int)
            for i, phase in enumerate(new_target):
                im, idxs = self.join_images(target, inten, idxs, phase)
                new_inten[i] = im
            target = new_target
            inten = np.array(new_inten).T
        
        target = np.array(target)
        inten = np.array(inten)
        return [target, inten]
                              
                              
                              
                              
    def get_inten_of_images(self, names, path, filtered = None):
        images = []
        target = []
        for item in names:
            images.append(np.array(imageio.imread(str(path/item))))
            target.append(float(item.split("_")[1]))
        images = np.array(images)
        target = np.array(target)
        
        # filter the images
        if filtered:
            images = images[filtered(target)]
            target = target[filtered(target)]
        inten = np.array([item.mean() for item in images])
        return target, inten
                                  
                                  
    def make_final_target(self, target):
        minimum = np.array([item[0] for item in target]).min()
        lenths = np.array([len(item) for item in target]).min()
        result = np.array(target[np.where(lenths == lenths.min())[0][0]])
        result = result[np.where(result == minimum)[0][0]:]
        return result
                                 
                                 
    def get_average_data(self, target, images):
        new_target = []
        new_images = []
        idx = 0
        while idx < len(target):
            a, b, idx = self.get_items(target, images, idx)
            new_target.append(a)
            new_images.append(b)
        return np.array(new_target), np.array(new_images)
    
    def join_images(self, target, inten, idxs, phase):
        result = []
        for i, item in enumerate(target):
            while True:
                if idxs[i] >= len(item):
                    idxs[i] = 0

                if item[idxs[i]] != phase:
                    idxs[i] += 1
                else:
                    break
            result.append(inten[i][idxs[i]])
        result = np.array(result).flatten()

        return result, idxs
    
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
        
        
            