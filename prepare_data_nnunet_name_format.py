from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
           ###########  Dataloader  #############
import numpy as np
import SimpleITK as sitk
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
depth = 64 


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
           ###########  Dataloader  #############
import numpy as np
import SimpleITK as sitk
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
depth = 64 

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import nibabel as nib
import os

def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 
    
    
def resample_image(image: sitk.Image, out_spacing: Tuple[float] = (0.707, .707, 2.0),
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            out_size = (np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
        else:
            out_size = np.array(out_size)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(image)
 

class Dataset_V(Dataset): 
    def __init__(self, images_folder,transformations=None):
      
        self.images_folder = images_folder
        self.transformations = transformations
        self.images = os.listdir(images_folder)
    
    def __len__(self):
       return len(self.images)
   
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,self.images[index])
        b = self.images[index].rfind('(')
        if b == -1:
            img_path = os.path.join(img_path,self.images[index]+'.nrrd')
            gt_path =  os.path.join(img_path[:-5]+'.seg.nrrd')
        
        else :
            img_path = os.path.join(img_path,self.images[index][:b-1]+'.nrrd')
            gt_path =  os.path.join(img_path[:-5]+'.seg.nrrd')
                    
        img = sitk.ReadImage(img_path)
        img = resample_image(img,is_label=True) 
        img = sitk.GetArrayFromImage(img)
        img = img.astype(np.float64) 
        
        gt = sitk.ReadImage(gt_path)
        gt = resample_image(gt,is_label=True) 
        gt = sitk.GetArrayFromImage(gt)
        gt = gt.astype(np.float64)
        
       
        return img,gt,self.images[index]
        
def Data_Loader_V(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader



val_imgs = '/data/scratch/acw676/Seg_A/all_data/'
train_loader = Data_Loader_V(val_imgs,batch_size = 1)
a = iter(train_loader)
save_path_training = '/data/scratch/acw676/nn/data3/training/'

for i in range(56):
    a1 =next(a)
    name = 'SEGA' + '_'+str(i).zfill(3)
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
    to_format_img.to_filename(os.path.join(path,name+'.nii.gz'))
        
        ###  saving the gts 
    gt = np.moveaxis(gt, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_gt = nib.Nifti1Image(gt, np.eye(4))  
    to_format_gt.set_data_dtype(np.uint8)
    to_format_gt.to_filename(os.path.join(path,name+'_gt'+'.nii.gz'))
