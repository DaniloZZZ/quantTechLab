#from torchvision import datasets, transforms
import torch
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import imageio
import random




class train_test_split():
    def __init__(self, folder = None,
                 filtered = None, transform = None, pca = None, paths = None):
        """
        pca: get module PCA(N_components).
        
        
        """
        self.transform = transform
        self.filtered = filtered
        self.pca = pca
        self.folder = Path(folder)
        self.names = os.listdir(self.folder)
        need = lambda x: ".png" in x
        self.names = list(filter(need, self.names))
        self.names.sort(key= lambda x: int(x.split("_")[0]))
    
    def get(self, start = 0, end = 1, shuffle = False, test_size = 0.2):
        self.names = self.names[int(start*len(self.names)):int(end*len(self.names))]  
        if shuffle:
            random.shuffle(self.names)
        train_names = self.names[:int((1-test_size)*len(self.names))]
        test_names = self.names[int((1-test_size)*len(self.names)):]
        train_dataset = FolderImageDataset(self.folder, filtered = self.filtered, 
                                          transform = self.transform, names = train_names) 
        test_dataset = FolderImageDataset(self.folder, filtered = self.filtered, 
                                          transform = self.transform, names = test_names)
        if self.pca:
                self.pca.fit(self.data_transform(train_dataset))
                train_dataset.images = np.array(self.pca.transform(self.data_transform(train_dataset)), dtype = float)
                test_dataset.images = np.array(self.pca.transform(self.data_transform(test_dataset)), dtype = float)
        return train_dataset, test_dataset
    
    
    def data_transform(self, data):
        return np.array(data.images).reshape(len(data), -1)


class FolderImageDataset(Dataset):
    def __init__(self, folder,
                 filtered = None, transform = None, names = None):
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
        self.folder = Path(folder)
        if names:
            self.names = names
        else:
            
            
            self.names = os.listdir(self.folder)[:-1]
            need = lambda x: ".png" in x
            self.names = list(filter(need, self.names))
            self.names.sort(key= lambda x: int(x.split("_")[0]))
        #! TODO: adapt to new parameter interface
#         size = end-start
#         start = end==1
#         #---------
#         #self.names = self.names[:int(0.5*(len(self.names)))]
        
#         if start:
#             self.names = self.names[:int(size*(len(self.names)))]
#         else: 
#             self.names = self.names[int((1-size)*(len(self.names))): ]
#         self.names.sort(key= lambda x: int(x.split("_")[0]))
        self.transform = transform
        self.images = []
        self.target = []
        for item in self.names:
            self.images.append(np.array(imageio.imread(str(self.folder/item))))
            self.target.append(float(item.split("_")[1]))
        self.images = np.array(self.images)
        self.target = np.array(self.target)
        
        # filter the images 
        if filtered is not None:
            self.images = self.images[filtered(self.target)]
            self.target = self.target[filtered(self.target)]
        
        
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
    
    def print_target_statistic(self):
        print(f"MEAN = {self.target.mean()} \t MSE = {self.target.std()**2} \t SIGMA = {self.target.std()}")