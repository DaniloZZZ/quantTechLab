import os
import numpy as np
import imageio

class dataset_for_classic():
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
    
    def join_images_inten(self, target, inten, idxs, phase):
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ## FOR PCA DATASET
    
    
    
import torch
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import imageio
import random
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
 

class train_test_for_pca_and_classic():
    def __init__(self, paths,
                  N_COMPONENTS, filtered = None, transform = None, average = False):
        """
        pca: get module PCA(N_components).
        
        
        """
        self.average = average
        self.components = N_COMPONENTS
        self.transform = transform
        self.filtered = filtered
        self.paths = paths
        self.names = []
        for path in self.paths:
            self.names.append(self.get_valid_names(path))
        
        self.names = np.array(self.names)
    
    def get(self, start = 0, end = 1, shuffle = False, test_size = 0.2):
        
        train_names = []
        test_names = []
        
        for item in self.names:
            train, test = self.split(item, start = start, end = end, shuffle = shuffle, test_size = test_size)
            train_names.append(train)
            test_names.append(test)
        
        train_dataset = FolderImageDataset(self.paths, filtered = self.filtered, 
                                          transform = self.transform, names = train_names, N_COMPONENTS = self.components) 
        test_dataset = FolderImageDataset(self.paths, filtered = self.filtered, 
                                          transform = self.transform, names = test_names, N_COMPONENTS = self.components, pca = train_dataset.pcas)
        
        return train_dataset, test_dataset
    
    def split(self, names, shuffle = False, test_size = 0.2, start = 0, end = 1):
        names = np.array(names)
        idx = np.linspace(start*(len(names)-1),end*(len(names)-1), len(names), dtype = int)
        if shuffle:
            random.shuffle(idx)
        
        return names[idx[:int((1-test_size)*len(idx))]], names[idx[:int((test_size)*len(idx))]]
        
    def get_valid_names(self,path):
        names = os.listdir(path)
        
        need = lambda x: ".png" in x
        names = list(filter(need, names))
        
        names.sort(key= lambda x: int(x.split("_")[0]))
        
        k = 0
        
        for i in range(len(names)):
            if names[i+1].split("_")[1] < names[i].split("_")[1]:
                k = i
                break
       
        names = names[k+1:]
        for i in range(len(names)-1, 0, -1):
            if names[i-1].split("_")[1] > names[i].split("_")[1]:
                k = i
                break
        
        names = names[:k-1]
        return names
    
    
        
   
    
    
    
    
    
class FolderImageDataset(Dataset):
    def __init__(self, paths,
                 filtered = None, transform = None, names = None, N_COMPONENTS = 150, pca = None, average = False):
        """
        Reads images from folder and optionally applies a transform to them.
        Filter parameter takes target value and returns bool

        Parameters
        ----------
        folder : str
        start: float
            rel index of start. float from 0 to 1.
        end: float
            rel index of end. float from 0 to 1.
        transfrom : callable
            lazily apply transform to image.
        filtered: callable
            float ->  bool. filters the dataset based on target
        """
        
        self.average = average
        self.names = names
        self.paths = paths
        self.N_COMPONENTS = N_COMPONENTS
    
        self.transform = transform
        self.images = []
        self.target = []
        self.inten = []
        if pca:
            self.pcas = pca
            index = 0
        else:
            index = None
            self.pcas = []
        
        for path, names in zip(self.paths, self.names):
            target, images, inten,  pca_0 = self.get_pca_of_images(self.N_COMPONENTS, names, path, filtered = filtered, pca = index)
            self.target.append(target)
            self.images.append(images)
            self.inten.append(inten)
            if not pca:
                self.pcas.append(pca_0)
            else :
                index += 1
       
            
            
        new_target = self.make_final_target(self.target)
        new_images = np.zeros((len(new_target), len(self.target)*self.N_COMPONENTS))
        new_inten = np.zeros((len(new_target), len(self.target)))
        idxs = np.zeros(len(self.target), dtype = int)
        for i, phase in enumerate(new_target):
            im, idxs = self.join_images(self.target, self.images, idxs, phase)
            new_images[i] = im
            
            
        idxs = np.zeros(len(self.target), dtype = int)    
        for i, phase in enumerate(new_target):
            im, idxs = self.join_images_inten(self.target, self.inten, idxs, phase)
            new_inten[i] = im
        self.target = new_target
        self.images = new_images
        self.inten = new_inten.T
        
        
    def __getitem__(self, idx):
        
        np_img = self.images[idx]
        y_true = self.target[idx]
    
        if self.transform:
            np_img = self.transform(np_img)
            y_true = torch.tensor(y_true, dtype=torch.float)
            y_true = y_true.view(-1,)
        else:
            np_img = torch.tensor(np_img, dtype = torch.float)
            y_true = torch.tensor(y_true, dtype=torch.float)
            y_true = y_true.view(-1,)
        return np_img, y_true
 
    def __len__(self):
        return  len(self.target)
    
    
    def get_pca_of_images(self,N_COMPONENTS, names, path, filtered = None, pca = None):
        images = []
        target = []
        inten = []
        for item in names:
            im = np.array(imageio.imread(str(path/item)))
            inten.append(im.mean())
            images.append(im)
            target.append(float(item.split("_")[1]))
        images = np.array(images)
        target = np.array(target)
        inten = np.array(inten)
        
        # filter the images
        if filtered:
            images = images[filtered(target)]
            inten = inten[filtered(target)]
            target = target[filtered(target)]
        if not pca:
            pca = PCA(N_COMPONENTS)
            pca.fit(self.data_transform(images))
        else:
            pca = self.pcas[pca]
        images = np.array(pca.transform(self.data_transform(images)), dtype = float)
        return target, images, inten, pca
    
    def data_transform(self, images):
        return np.array(images).reshape(len(images), -1)  
    
    def print_target_statistic(self):
        print(f"MEAN = {self.target.mean()} \t MSE = {self.target.std()**2} \t SIGMA = {self.target.std()}")
        
    def get_items(self,target, images, idx):
        a = target[idx]
        b = np.zeros(len(images[idx]))
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
    
    def join_images(self, target, images, idxs, phase):
        result = []
        for i, item in enumerate(target):
            while True:
                if idxs[i] >= len(item):
                    idxs[i] = 0
                
                if item[idxs[i]] != phase:
                    idxs[i] += 1
                else:
                    break
            result.append(images[i][idxs[i]])
        result = np.array(result).flatten()
        
        return result, idxs
    
    
    
    
    def make_final_target(self, target):
        minimum = np.array([item[0] for item in target]).min()
        lenths = np.array([len(item) for item in target]).min()
        result = np.array(target[np.where(lenths == lenths.min())[0][0]])
        result = result[np.where(result == minimum)[0][0]:]
        return result
        
    
    def join_images_inten(self, target, inten, idxs, phase):
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
    
    