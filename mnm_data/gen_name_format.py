import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import SimpleITK as sitk
import cv2
from typing import List, Union, Tuple

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256

 
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
        ## SA_ES_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA_ES = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        
        ## SA_ES_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        temp_SA_ES = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        
        ## SA_ED_img ####
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA_ED = sitk.GetArrayFromImage(img)   ## --> [C,H,W]

        ## SA_ED_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img = sitk.ReadImage(img_SA_gt_path)
        temp_SA_ED = sitk.GetArrayFromImage(img)   ## --> [C,H,W]

        

        return img_SA_ES,temp_SA_ES,img_SA_ED,temp_SA_ED,self.images_name[index]
        
def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


val_imgs = r"C:\My_Data\M2M Data\data\data_2\train"
val_csv_path = r"C:\My_Data\M2M Data\data\train.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)


import nibabel as nib

save_path_training = r'C:\My_Data\M2M Data\data\data_2\data4'

for i in range(160):
    a1 =next(a)
    n = a1[4][0].numpy()
    name =str(n).zfill(3)
    path = os.path.join(save_path_training,name)
    os.mkdir(path)

    img = a1[0][0,:].numpy()
    gt = a1[1][0,:].numpy()
    
    if img.shape[0]!=gt.shape[0]:
      temp  =np.zeros(gt.shape)
      temp = img[0:img.shape[0]-1,:]
      img = temp
    
        ###  saving the imgs 
    img = np.moveaxis(img, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_img = nib.Nifti1Image(img, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(path,name+'_SA_ES'+'.nii.gz'))
        
        ###  saving the gts 
    gt = np.moveaxis(gt, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_gt = nib.Nifti1Image(gt, np.eye(4))  
    to_format_gt.set_data_dtype(np.uint8)
    to_format_gt.to_filename(os.path.join(path,name+'_SA_ES_gt'+'.nii.gz'))
    
    
    img = a1[2][0,:].numpy()
    gt = a1[3][0,:].numpy()
    
    if img.shape[0]!=gt.shape[0]:
      temp  =np.zeros(gt.shape)
      temp = img[0:img.shape[0]-1,:]
      img = temp
    
        ###  saving the imgs 
    img = np.moveaxis(img, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_img = nib.Nifti1Image(img, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(path,name+'_SA_ED'+'.nii.gz'))
        
        ###  saving the gts 
    gt = np.moveaxis(gt, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_gt = nib.Nifti1Image(gt, np.eye(4))  
    to_format_gt.set_data_dtype(np.uint8)
    to_format_gt.to_filename(os.path.join(path,name+'_SA_ED_gt'+'.nii.gz'))
