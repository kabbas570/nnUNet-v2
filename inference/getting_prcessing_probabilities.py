import numpy as np
from typing import Union
import SimpleITK as sitk
import pickle
import  torch





# pred2 = np.load(r"C:\My_Data\Hadi\P_Results\SEGA_006.npz")
# pred2 = pred2.f.probabilities
# pred2 = torch.tensor(pred2)
# pred2 = torch.argmax(pred2, dim=0)
# pred2 = np.array(pred2)

# pred1 = sitk.ReadImage(r"C:\My_Data\Hadi\P_Results\SEGA_006.nii.gz")
# pred1 = sitk.GetArrayFromImage(pred1)


# gt = sitk.ReadImage(r"C:\My_Data\Hadi\testing\SEGA_006\SEGA_006_gt.nii.gz")
# gt = sitk.GetArrayFromImage(gt)


# single_1 = (2 * (pred2 * pred1).sum()) / (
#                 (pred2 + pred1).sum() + 1e-8)

# single_2 = (2 * (pred2 * gt).sum()) / (
#                 (pred2 + gt).sum() + 1e-8)

# single_3 = (2 * (pred1 * gt).sum()) / (
#                 (pred1 + gt).sum() + 1e-8)





import numpy as np
import numpymaxflow
import torch
import nibabel as nib
import os
import glob

def graph_cut(image,pred,path,name):
    
    
    #pred = 1-pred # the pred probability should have [BG, FG] but the provided has [FG, BG] so 1-pred to convert to desired
    regularized = numpymaxflow.maxflow(image, pred, lamda=1, sigma=0.1, connectivity=18)
        
    img1 = regularized[0,:]
    
    img1 = np.moveaxis(img1, [0, 1, 2], [-1, -2, -3])   ## reverse the dimenssion sof array
    to_format_img = nib.Nifti1Image(img1, np.eye(4))  
    to_format_img.set_data_dtype(np.uint8)
    to_format_img.to_filename(os.path.join(path,name +'.nii.gz'))
    
    
#pred = np.load("/data/scratch/acw676/nn_unet_data/dataset/P_Results/SEGA_006.npz")
#pred = pred.f.probabilities
#
#img = sitk.ReadImage("/data/scratch/acw676/nn/data2/testing/SEGA_006/SEGA_006.nii.gz")
#img = sitk.GetArrayFromImage(img)
#img = np.expand_dims(img, axis=0)
#
#_= graph_cut(img,pred,'/data/scratch/acw676/graph_results/','SEGA_006')
#
#    
    
prob_id = []
for infile in sorted(glob.glob("/data/scratch/acw676/nn_unet_data/dataset/P_Results/*.npz")): 
    prob_id.append(infile)

for i in range(11):
    pred = np.load(prob_id[i])
    pred = pred.f.probabilities
    name = prob_id[i][-12:]
    name = name[:-4]
    img_path = "/data/scratch/acw676/nn/data2/testing/"+name+ '/'+ name+'.nii.gz'
    
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    img = np.expand_dims(img, axis=0)

    _= graph_cut(img,pred,'/data/scratch/acw676/graph_results/',name)
   
