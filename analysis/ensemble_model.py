import os
import time
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import imageio
from ImageDataset import FolderImageDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from Datagen import train_test
# from Compile_the_data import train_test_for_pca
from fitting_data import gauss_est
from fitting_data import data_from_dataset
from fitting_data import dataset_from_folder
from Compile_the_data import dataset_for_classic

import sklearn
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb



    
    
class ImageDataset(Dataset):
    def __init__(self):
        
        self.images = []
        self.target = []
        self.transform = 0
        
        
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




class Net_pcas(nn.Module):
    def __init__(self, N):
        super(Net_pcas, self).__init__()
        first = N
        second = int(N/2)
        third = int(N/4)
        self.linear1 = nn.Linear(first,second)
        self.linear2 = nn.Linear(second, third)
        self.linear3 = nn.Linear(third, 1)
        self.bn1 = nn.BatchNorm1d(int(N/2))
    def forward(self, x):
        
        y1 = F.sigmoid(self.linear1(x))
        #y1 = self.bn1(y1)
        y1 = F.sigmoid(self.linear2(y1))
        y1 = self.linear3(y1)
        
      
        return y1
    
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t mse: {}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), train_loss, target.std()**2))
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss

    test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.7f}\n'.format(
#         test_loss))
    return test_loss





params_net = {"max_size": 1,
              "min_size": 1,
              "min_features" : 0.8,
              "max_features" : 1,
              "max_epoches" : 300,
              "min_epoches" : 200,
              "Batches" : [5, 3],
              "lr" : [1e-3, 1e-4]
}

params_classic = {"max_size": 1,
                  "min_size": 0.5,
                  "h" : [ 0.0001, 0.0002, 0.5*0.0001]
}

params_tree = {"max_size": 1,
               "min_size": 0.5,
               "min_features" : 0.5,
               "max_features" : 1,
               "lr" : [1.01, 1, 0.9], 
               "max_depth" : [80, 100], 
               "n_estimators" : [100, 150, 50]
}


class ensemble_models():
    def __init__(self, n_classics = 0, n_neural = 0, n_forests = 0, 
                 params_net = params_net, params_classic = params_classic, params_tree = params_tree, start = 0.4, end = 0.6):
        self.n_classics = n_classics
        self.n_neural = n_neural
        self.n_forests = n_forests
        self.params_net = params_net
        self.params_classic = params_classic
        self.params_tree = params_tree
        self.idxes = []
        self.start = start
        self.end = end
        
    def fit(self, classic_data = None, net_data = None):
        self.models = []
        if self.n_classics:
            print("Starting classical fitting\n")
            start_time = time.time()
            for _ in range(self.n_classics):
                target = classic_data[0]
                inten = classic_data[1]
                size_of_data = self.params_classic["min_size"] + random.random()*(
                self.params_classic["max_size"] - self.params_classic["min_size"])
                
                target, inten = self.get_part_of_data(size_of_data, target, inten)
                
                h = random.choice(self.params_classic["h"])
                
                model = gauss_est(h)
                model.fit(target, inten)
                self.models.append(model)
            end_time = time.time()
            print("Finished classical fitting:\n It took " + self.get_time(end_time - start_time) + "\n")
    
    
    
        if self.n_neural:
            print("Starting machine learning\n")
            start_time = time.time()
            
            self.Batches = []
            for o in range(self.n_neural):
                
                print("Starting train {} model\n".format(o+1))
                start_time_1 = time.time()
                dataset, N = self.get_part_of_dataset(net_data, "net")
                
                
                BATCH = random.choice(self.params_net["Batches"])
                self.Batches.append(BATCH)
                lr = random.choice(self.params_net["lr"])
                epoches = int(self.params_net["min_epoches"] + random.random()*(
                self.params_net["max_epoches"] - self.params_net["min_epoches"]))
                
                loader = torch.utils.data.DataLoader(
                        net_data,
                     batch_size=BATCH, shuffle=True)
                model = Net_pcas(N)
                use_cuda = False
                device = torch.device("cuda" if use_cuda else "cpu")
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                
                for epoch in range(1, epoches):
                    
                    los  = train( model, device, loader, optimizer, epoch)
                    los = test( model, device, loader)
                    
                    if los == None:
                        print("BATCH = {} \t lr = {} \t epoches = {} \t N = {}\n".format(BATCH, lr, epoches, N))
                    
                self.models.append(model)
                end_time_1 = time.time()
                print("---------------------------------------------\n")
                print("Finished train {} model. It took ".format(o+1) + self.get_time(end_time_1 - start_time_1) + "\n")
            end_time = time.time()
            print("Finished machine learning:\n It took " + self.get_time(end_time - start_time) + "\n")
            
            
            
            
        if self.n_forests:
            
            print("Starting Random forest and XGBOOST\n")
            start_time = time.time()
            
            for _ in range(self.n_forests):
                dataset, N = self.get_part_of_dataset(net_data, "tree")
                
                
                lr = random.choice(self.params_tree["lr"])
                n_estimators = random.choice(self.params_tree["n_estimators"])
                max_depth = random.choice(self.params_tree["max_depth"])
                
                if False:
                    estimator = xgb.XGBRegressor(learning_rate=lr, max_depth=max_depth, n_estimators=n_estimators, min_child_weight=5)
                else: 
                    estimator = ensemble.RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = int(self.params_tree["max_features"]*N))
                
                estimator.fit(dataset.images, dataset.target)
                self.models.append(estimator)
                
            end_time = time.time()
            print("Finished Random forest and XGBOOST:\n It took " + self.get_time(end_time - start_time) + "\n")
            
                
                
                
                
                
      
    
    def predict(self, classic_data = None, net_data = None):
        prediction = []
        for i in range(self.n_classics):
            X_predicted,_, prob_result = self.models[i].predict(classic_data)
            prediction.append(X_predicted)
    
        for i in range(self.n_neural):
            BATCH = 5
            dataset = ImageDataset()
            dataset.target = net_data.target
            dataset.images = net_data.images[:, self.idxes[i]]
            dataset.transform = net_data.transform
            loader = torch.utils.data.DataLoader(
                        dataset,
                     batch_size=BATCH, shuffle=False)
            
            pred_np = np.array([])
            for x,_ in loader:
                pred = self.models[i + self.n_classics](x)
                pred_np = np.concatenate((pred_np,np.array(pred.reshape(-1).detach())))
            prediction.append(pred_np)
            
        for i in range(self.n_forests):
            
            dataset = ImageDataset()
            dataset.target = net_data.target
            dataset.images = net_data.images[:, self.idxes[self.n_neural+i]]
            dataset.transform = net_data.transform
            predicted = self.models[i + self.n_neural + self.n_classics].predict(dataset.images)
            prediction.append(predicted)
            
            
            
        return prediction, self.make_prediction(prediction)
                
    
    
    
    def get_part_of_data(self, size, target, images):
        new_target = []
        new_images = []
        
        for i in range(len(images)):
            idx = np.linspace(0, len(images[i])-1, len(images[i]), dtype = int)
            random.shuffle(idx)
            idx = idx[:int(size*len(idx))]
            
            new_target.append(target[i][idx])
            new_images.append(images[i][idx])
        return np.array(new_target), np.array(new_images)
            
     
    def get_part_of_dataset(self, dataset, point):
        if point == "net":
            params = self.params_net
        else: 
            params = self.params_tree
        
        
        
        result_data = ImageDataset()
        idx1 = np.linspace(0, len(dataset.images)-1, len(dataset.images), dtype = int)
        
        size = params["min_size"] + random.random()*(
        params["max_size"] - params["min_size"])
        
        if size != 1:
            random.shuffle(idx1)
        
            idx1 = idx1[:int(size*len(idx1))]



        idx2 = np.linspace(0, len(dataset.images[0])-1, len(dataset.images[0]), dtype = int)
        
        
        
        
        size_features = params["min_features"] + random.random()*(
        params["max_features"] - params["min_features"])
        
        if size != 1:
            random.shuffle(idx2)
            idx2 = idx2[:int(size_features*len(idx2))]
        
        result_data.transform = dataset.transform
        result_data.target = dataset.target[idx1]
        
        result_data.images = dataset.images[idx1]
        result_data.images = result_data.images[:,idx2]

        self.idxes.append(idx2)
        N = len(idx2)
#         print(result_data.target.shape)
#         print(result_data.images.shape)
        return result_data, N
                
                
            
    def make_prediction(self, prediction):
        final = np.zeros(len(prediction[0]))
        prediction = np.array(prediction)
        size = len(prediction[0])
        prediction = prediction.T
        for i in range(size):
            final[i] = self.clear_ans(prediction[i])
        
        return final
                
      
    def clear_ans(self, ans):
        result = ans
        result.sort()
        result = result[int(self.start*len(result)):int(self.end*len(result))]
        return result.mean()
                
                
                
    def get_time(self, time):
        s = "'"+ str(int(time*100) % 100)
        s = str(int(time) % 60) + s
        time = int(time) // 60
        while time != 0:
            s = str(time % 60) + ":" + s
            time = time // 60
        return s             

        
        
        
        
        