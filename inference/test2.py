import cc3d
import SimpleITK as sitk
import glob
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

               
          #### Set these two paths only    ###############
          
          #############################
                  ###############  
                    ######

input_folder = '/data/scratch/acw676/Nay_Data/Data_from_Nay/imgs/'  ## path to folder where we have .nrrrd files 
output_folder = '/data/scratch/acw676/Nay_Data/Data_from_Nay/preds/'            ## path to folder where we we want to save the predictions


            #############################
                  ###############  
                    ######
                    
def func1():
    files = os.listdir(input_folder)
    image_files = [f for f in files if f.lower().endswith(('.nrrd'))]
    
    for i, filename in enumerate(image_files):
        # Construct the new filename with suffix 0000
        new_filename = f"{os.path.splitext(filename)[0]}_{0:04d}{os.path.splitext(filename)[1]}"
        old_path = os.path.join(input_folder, filename)
        new_path = os.path.join(input_folder, new_filename)
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
     
 
    

def func2():
    paths_ = []
    for infile in sorted(glob.glob(output_folder + '*.nrrd')):
        paths_.append(infile)
    
    for i in range(len(paths_)):
        
        name = paths_[i][len(output_folder):] 
        pred_itk = sitk.ReadImage(paths_[i])
        pred = sitk.GetArrayFromImage(pred_itk)
        print(pred.shape)
        pred, _ = cc3d.largest_k(
          pred, k=1, 
          connectivity=18, delta = int(0),
          return_N=True,
        )
        
        pred = sitk.GetImageFromArray(pred)
        pred.CopyInformation(pred_itk)
        sitk.WriteImage(pred,output_folder+name)
    

def func3():
    paths_ = []
    for infile in sorted(glob.glob(output_folder + '*.nrrd')):
        paths_.append(infile)
    
    for i in range(len(paths_)):
        
        name = paths_[i][len(output_folder):] 
        pred_itk = sitk.ReadImage(paths_[i])
        pred = sitk.GetArrayFromImage(pred_itk)
        
        pred = sitk.GetImageFromArray(pred)
        pred.CopyInformation(pred_itk)
        sitk.WriteImage(pred,output_folder+name[:-5] + '_seg' + '.nrrd')
        

def run ():
    #_ = func1()
    _ = predict()
    #_ = func2()
    _ = func3()
 

if __name__ == "__main__":
    run()
