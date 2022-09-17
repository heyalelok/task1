from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import torch.nn.functional as F


class cifar_dataset(Dataset): 
    def __init__(self, transform, mode): 
        self.transform = transform
        self.mode = mode  
     
        if self.mode=='test':
            self.test_data = []
            with open('./data/test_label1.txt','r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = './data/test/%s' % (entry[0])
                    self.test_data.append(img_path)

        else:
            train_data = []
            noise_label = []
            with open('./data/label.txt', 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()           
                    img_path = './data/train/%s' % (entry[0])
                    train_data.append(img_path)
                    noise_label.append( int(entry[1]) ) 

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label

    def __getitem__(self, index):
        if self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.open(img).convert('RGB')
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img = self.test_data[index]
            img = Image.open(img).convert('RGB')
            img = self.transform(img)            
            return img
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

 
    def run(self,mode,losses=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='test':
            test_dataset = cifar_dataset(transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader


