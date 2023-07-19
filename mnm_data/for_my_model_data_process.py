import os
import SimpleITK as sitk
import numpy as np

images_folder = r'C:\My_Data\M2M Data\data\data_2\train'

imgs_path = r'C:\My_Data\M2M Data\data\nnunet_data\train\imgs/'
gts_path = r'C:\My_Data\M2M Data\data\nnunet_data\train\gts/'


# for i in range(1,161):
#     img_path = os.path.join(images_folder, str(i).zfill(3),str(i).zfill(3)) +'_LA_ES.nii.gz'
#     img = sitk.ReadImage(img_path)
#     name  = img_path[42:-7]+'_0000' + '.nii.gz'
#     sitk.WriteImage(img,imgs_path+name)
    
#     img_path = os.path.join(images_folder, str(i).zfill(3),str(i).zfill(3)) +'_LA_ED.nii.gz'
#     img = sitk.ReadImage(img_path)
#     name  = img_path[42:-7]+'_0000' + '.nii.gz'
#     sitk.WriteImage(img,imgs_path+name)
    
#     img_path = os.path.join(images_folder, str(i).zfill(3),str(i).zfill(3)) +'_LA_ES_gt.nii.gz'
#     img = sitk.ReadImage(img_path)
#     name  = img_path[42:]
#     sitk.WriteImage(img,gts_path+name)
    
#     img_path = os.path.join(images_folder, str(i).zfill(3),str(i).zfill(3)) +'_LA_ED_gt.nii.gz'
#     img = sitk.ReadImage(img_path)
#     name  = img_path[42:]
#     sitk.WriteImage(img,gts_path+name)
