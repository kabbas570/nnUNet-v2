import os 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import os

input_folder = '/data/DERI-CDTwins/Segmentation/input_imgs/tavi/TAVI001/TAVI001/CT_nrrds/'

output_folder = '/data/DERI-CDTwins/Segmentation/input_imgs/tavi/TAVI001_pred/'


import os
from PIL import Image

def func1(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only image files (you can add more extensions if needed)
    image_files = [f for f in files if f.lower().endswith(('.nrrd'))]
    
    for i, filename in enumerate(image_files):
        # Construct the new filename with suffix
        new_filename = f"{os.path.splitext(filename)[0]}_{0:04d}{os.path.splitext(filename)[1]}"
        
        # Generate full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        
def predict():
    # Define variables
    
     #Model 1 prediction
     predictor = nnUNetPredictor(
         tile_step_size=0.5,
         use_gaussian=True,
         use_mirroring=True,
         perform_everything_on_gpu=True,
         device=torch.device('cuda', 0),
         verbose=True,
         verbose_preprocessing=True,
         allow_tqdm=True
     )

     

     predictor.initialize_from_trained_model_folder(
        '/data/DERI-CDTwins/Segmentation/algorithm/weights/Dataset050_SEGA/nnUNetTrainer__nnUNetPlans__3d_fullres/',
         use_folds=(0,1,2,3,4),
         #use_folds=(3,),
         checkpoint_name='checkpoint_final.pth',
     )
     
     
     predictor.predict_from_files(input_folder,
                            output_folder,
                            save_probabilities=False, overwrite=True,
                            num_processes_preprocessing=1, num_processes_segmentation_export=1,
                            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)   
     
    

def run ():
    _ = func1(input_folder)
    _ = predict()
 

if __name__ == "__main__":
    run()