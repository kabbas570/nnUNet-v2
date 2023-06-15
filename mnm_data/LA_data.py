import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import SimpleITK as sitk
import cv2
from typing import List, Union, Tuple
import nibabel as nib
           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
save_path_training = r'C:\My_Data\M2M Data\data\data_2\testing'

class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        
        name = str(self.images_name[index]).zfill(3)
        
        path = os.path.join(save_path_training,name)
        os.mkdir(path)
        
        path_img = img_path+'_LA_ES.nii.gz'
        img = nib.load(path_img)
        img.to_filename(os.path.join(path,name+'_LA_ES'+'.nii.gz'))
        
        path_img = img_path+'_LA_ES_gt.nii.gz'
        img = nib.load(path_img)
        img.to_filename(os.path.join(path,name+'_LA_ES_gt'+'.nii.gz'))
        
        path_img = img_path+'_LA_ED.nii.gz'
        img = nib.load(path_img)
        img.to_filename(os.path.join(path,name+'_LA_ED'+'.nii.gz'))
        
        path_img = img_path+'_LA_ED_gt.nii.gz'
        img = nib.load(path_img)
        img.to_filename(os.path.join(path,name+'_LA_ED_gt'+'.nii.gz'))
        
        return 0
        
        
def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


val_imgs = r"C:\My_Data\M2M Data\data\data_2\val"
val_csv_path = r"C:\My_Data\M2M Data\data\val.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)


for i in range(40):
    a1 =next(a)
