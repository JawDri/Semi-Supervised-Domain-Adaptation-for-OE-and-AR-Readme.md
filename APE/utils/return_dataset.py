import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist
import torch.utils.data as util_data
import pandas as pd
import numpy as np


Source_train = pd.read_csv("/content/drive/MyDrive/APE/data/Source_train.csv")
Source_test = pd.read_csv("/content/drive/MyDrive/APE/data/Source_test.csv")
Target_train = pd.read_csv("/content/drive/MyDrive/APE/data/Target_train.csv")
Target_train_unl = pd.read_csv("/content/drive/MyDrive/APE/data/Target_train_unl.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/APE/data/Target_test.csv")
Target_val = pd.read_csv("/content/drive/MyDrive/APE/data/Target_val.csv")

FEATURES_dset = list(i for i in Source_train.columns if i!= 'labels')
len_features = len(FEATURES_dset)
unique_labels = Source_train['labels'].unique().tolist()

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


Source_train_dset = PytorchDataSet(Source_train, len_features)
Source_test_dset = PytorchDataSet(Source_test, len_features)
Target_train_dset = PytorchDataSet(Target_train, len_features)
Target_test_dset = PytorchDataSet(Target_test, len_features)
Target_train_unl_dset = PytorchDataSet(Target_train_unl, len_features)
Target_val_dset = PytorchDataSet(Target_val, len_features)

def return_dataset(args):

    

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
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
    }



    source_dataset = Source_train_dset
    target_dataset = Target_train_dset
    target_dataset_val =Target_val_dset
    target_dataset_unl = Target_train_unl_dset
    target_dataset_test =Target_test_dset
    class_list = unique_labels



    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 64
    else:
        bs = 64
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=1, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=1,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=1,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs*2, num_workers=1,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs, num_workers=1,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list

def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
        
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 24
    else:
        bs = 16
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
