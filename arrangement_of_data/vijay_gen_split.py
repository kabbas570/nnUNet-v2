import SimpleITK as sitk
import numpy as np
import os
import SimpleITK as sitk

# Path to your .nii.gz file
file_path = r"C:\My_Data\M2M Data\data\data_2\five_fold\data_la\data0\training\129\129_LA_ED_gt.nii.gz"
# Convert the SimpleITK image to a numpy array

# Read the image
image = sitk.ReadImage(file_path)
image_array = sitk.GetArrayFromImage(image)
image_array=image_array[0,:]


# Get image size
size = image.GetSize()
print("Image size:", size)

# Get image spacing (voxel size)
spacing = image.GetSpacing()
print("Voxel size:", spacing)

# Get image origin
origin = image.GetOrigin()
print("Image origin:", origin)

# Get image direction (orientation)
direction = image.GetDirection()
print("Image direction:", direction)



import numpy as np

# Path to your .npz file
file_path = r"C:\My_Data\M2M Data\data\data_2\five_fold\241_LA_ED.npz"

# Load the .npz file without reading the arrays into memory
with np.load(file_path, allow_pickle=True) as data:
    # Get the list of array names
    array_names = data.files

# Print the array names
print("Arrays in the .npz file:")
for name in array_names:
    print(name)
    
import numpy as np

# Path to your .npz file

# Load the .npz file
with np.load(file_path) as data:
    # Get the 'seg' array
    seg_array = data['seg']
def resample_itk_image_LA(itk_image):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing = (1,1,1)
    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image



path = r'C:\My_Data\M2M Data\data\data_2\five_fold\data_la\data0\training/'


# Create a mask for labels 1, 2, and 3

# Set all other labels to zero


# If you need to convert it back to a SimpleITK image



for i in range(1,361):
    
    path_img_ES = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ED.nii.gz'
    image = sitk.ReadImage(path_img_ES)
    image = resample_itk_image_LA(image)
    sitk.WriteImage(image, path_img_ES)
    
    
    path_img_ED = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ES.nii.gz'
    image = sitk.ReadImage(path_img_ED)
    image = resample_itk_image_LA(image)
    sitk.WriteImage(image, path_img_ED)
    
    path_gt_ES = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ED_gt.nii.gz'
    GT = sitk.ReadImage(path_gt_ES)
    GT = resample_itk_image_LA(GT)
    GT = sitk.GetArrayFromImage(GT)
    mask = np.isin(GT, [1, 2, 3])
    GT[~mask] = 0
    GT = sitk.GetImageFromArray(GT)
    sitk.WriteImage(GT, path_gt_ES)
    
    path_gt_ED = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ES_gt.nii.gz'
    GT = sitk.ReadImage(path_gt_ED)
    GT = resample_itk_image_LA(GT)
    GT = sitk.GetArrayFromImage(GT)
    mask = np.isin(GT, [1, 2, 3])
    GT[~mask] = 0
    GT = sitk.GetImageFromArray(GT)
    sitk.WriteImage(GT, path_gt_ED)
    
    
for i in range(1,361):
    
    path_img_ES = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ED.nii.gz'
    image = sitk.ReadImage(path_img_ES)
    spacing = image.GetSpacing()
    
    print("Voxel size:", spacing)
    
    
    path_img_ED = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ES.nii.gz'
    image = sitk.ReadImage(path_img_ED)
    spacing = image.GetSpacing()
    print("Voxel size:", spacing)
    
    path_gt_ES = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ED_gt.nii.gz'
    GT = sitk.ReadImage(path_gt_ES)
    spacing = GT.GetSpacing()
    print("Voxel size:", spacing)
    
    path_gt_ED = path + str(i).zfill(3) + '/' + str(i).zfill(3) + '_LA_ES_gt.nii.gz'
    GT = sitk.ReadImage(path_gt_ED)
    spacing = GT.GetSpacing()
    print("Voxel size:", spacing)
    
    
    
    
    
    
