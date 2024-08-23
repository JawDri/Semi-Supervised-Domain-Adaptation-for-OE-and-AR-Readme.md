import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist,Imagelists_VISDA_Test
from loaders.data_list import Imagelists_VISDA, return_classlist, Imagelists_VISDA_un
import torch.utils.data as util_data
import pickle
import pandas as pd
from pdb import set_trace as breakpoint
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random

from PIL import Image
import copy
import pdb
from torch.utils.data.dataloader import default_collate

from loaders.data_list import make_dataset_fromlist

from .randaugment import RandAugmentMC



Source_train_1 = pd.read_csv("/content/drive/MyDrive/CLDA/data/Source_train_1.csv")
Source_train_2 = pd.read_csv("/content/drive/MyDrive/CLDA/data/Source_train_2.csv")
Source_train_3 = pd.read_csv("/content/drive/MyDrive/CLDA/data/Source_train_3.csv")

Source_test = pd.read_csv("/content/drive/MyDrive/CLDA/data/Source_test.csv")
Target_train = pd.read_csv("/content/drive/MyDrive/CLDA/data/Target_train.csv")
Target_train_unl = pd.read_csv("/content/drive/MyDrive/CLDA/data/Target_train_unl.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/CLDA/data/Target_test.csv")
Target_val = pd.read_csv("/content/drive/MyDrive/CLDA/data/Target_val.csv")
FEATURES_dset = list(i for i in Source_train_1.columns if i!= 'labels')
len_features = len(FEATURES_dset)
unique_labels = Source_train_1['labels'].unique().tolist()

class PytorchDataSet(util_data.Dataset):
    
    def __init__(self, df, len_features):
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return self.train_X[idx].view(1, len_features), self.train_Y[idx]


class PytorchDataSetIndx(util_data.Dataset):
    
    def __init__(self, df, len_features):
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return self.train_X[idx].view(1, len_features), self.train_Y[idx], idx

class SubDataset(util_data.Dataset):
    def __init__(self, dataset,indexes):
        self.dataset = dataset
        self.len = len(indexes)
        self.indexes = indexes

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, target  = self.dataset[self.indexes[index]]
        return img,target,index


Source_train_dset_1 = PytorchDataSet(Source_train_1, len_features)
Source_train_dset_2 = PytorchDataSet(Source_train_2, len_features)
Source_train_dset_3 = PytorchDataSet(Source_train_3, len_features)
Source_test_dset = PytorchDataSet(Source_test, len_features)
Target_train_dset = PytorchDataSet(Target_train, len_features)
Target_test_dset = PytorchDataSet(Target_test, len_features)
Target_train_unl_dset = PytorchDataSet(Target_train_unl, len_features)
Target_val_dset = PytorchDataSet(Target_val, len_features)


def return_dataset_balance_self(args,test=False):



    #torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'train': transforms.Compose([

            transforms.ToTensor(),

        ]),
        'val': transforms.Compose([

            transforms.ToTensor(),

        ]),
        'test': transforms.Compose([

            transforms.ToTensor(),

        ]),
        'self': transforms.Compose([

            transforms.ToTensor(),

        ]),
    }

    dict_path2img = None


    train_dataset_1 = Source_train_dset_1
    train_dataset_2 = Source_train_dset_2
    train_dataset_3 = Source_train_dset_3
    target_dataset_val = Target_val_dset
    if test:
        target_dataset_test = Target_test_dset
    else:
        target_dataset_test = Target_test_dset
    target_dataset_unl = Target_train_unl_dset
    
    target_dataset = Target_train_dset
    class_list = unique_labels

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)    

    bs = 64

    nw = 1


    source_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=bs,
        num_workers=nw, shuffle=True, drop_last=True)
    source_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=bs,
        num_workers=nw, shuffle=True, drop_last=True)
    source_loader_3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=bs,
        num_workers=nw, shuffle=True, drop_last=True)
    labeled_target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=bs,
        num_workers=nw, shuffle=True, drop_last=True)

    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=nw, shuffle=True, drop_last=False)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs, num_workers=nw,
                                    shuffle=True, drop_last=False)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                batch_size=bs, num_workers=nw, shuffle=True, drop_last=True)

    return source_loader_1, source_loader_2, source_loader_3, labeled_target_loader, target_loader_val, target_loader_test, target_loader_unl, class_list

