import torch
import SimpleITK
#import numpy as np
import trimesh
from pathlib import Path

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import subprocess
#import shutil
import os
import gc

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class Segaalgorithm(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
           # input_path=Path('/input/'),
            input_path=Path('/input/images/ct/'),
            output_path=Path('/output/'),
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        print('this')
        print('input FILES ',os.listdir('/input/images/ct/'))
        #print('APP ',os.listdir('/opt/app/'))
        self._segmentation_output_path = self._output_path / "images" / "aorta-segmentation"
        if not self._segmentation_output_path.exists():
            self._segmentation_output_path.mkdir(parents=True)
        
        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/app/nnUNet/input")
        self.nnunet_out_dir = Path("/opt/app/nnUNet/output")
        self.nnunet_results = Path("/opt/app/results")
        self.nnunet_model1_out = Path("/opt/app/nnUNet/output/model_1")
        

        
        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_model1_out.mkdir(exist_ok=True, parents=True)
        



    def process_case(self, *, idx, case):
        # Load and test the image for this case
        print(case)
        print(idx)
        input_image, input_image_file_path = self._load_input_image(case=case)
        print('Image Name ',input_image_file_path.name)
        
        # Segment (and mesh)
        predictions = self.predict(input_image_path=input_image_file_path)
        #predictions = self.predict(input_image_path=input_image_file_path)
        aorta_segmentation = predictions[0]
        visualization_mesh = predictions[1]     # surface to be used for visualization
        volume_mesh = predictions[2]            # surface to be converted to volume mesh

        # Write resulting segmentation to output locations

        segmentation_path = self._segmentation_output_path / input_image_file_path.name
        visualization_path = self._output_path / "aortic-vessel-tree.obj"
        mesh_path = self._output_path / "aortic-vessel-tree-volume-mesh.obj"

        SimpleITK.WriteImage(aorta_segmentation, str(segmentation_path), True)
        trimesh.exchange.export.export_mesh(visualization_mesh, str(visualization_path), 'obj')
        trimesh.exchange.export.export_mesh(volume_mesh, str(mesh_path), 'obj')

        # Write segmentation file path to 'result.json' for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=segmentation_path.name),
                dict(type="wavefront", filename=visualization_path.name),
                dict(type="wavefront", filename=mesh_path.name)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_image_file_path.name)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_image_path: Path):# -> SimpleITK.Image:
        
        # Convert Image to nnUnet format
        print('Working on device: ',torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #name=input_image_path.name[:-4]+'_0000'+input_image_path.name[-4:]
        #ext=input_image_path.suffix
        input_image_path=str(input_image_path)

        output_image_path = str(self.nnunet_inp_dir / 'test_1_0000.nrrd')
        
        
        input_image=SimpleITK.ReadImage(input_image_path)
        
        SimpleITK.WriteImage(input_image, output_image_path)

        #shutil.copy(input_image_path, output_image_path)
        #print(input_image_path)
        #print('FILES NNUNET',os.listdir(str(self.nnunet_inp_dir)))
        
        #input_image,props=SimpleITKIO().read_images([input_image_path])
        
        # Define results folder

        os.environ['nnUNet_results']=str(self.nnunet_results)
        
        # Define variables
        input_folder=str(self.nnunet_inp_dir)

        model1_output=str(self.nnunet_model1_out)

        
        
        
        #Model 1 prediction
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        predictor.initialize_from_trained_model_folder(
            str(self.nnunet_results / 'Dataset050_SEGA/nnUNetTrainer__nnUNetPlans__3d_fullres'),
           use_folds=(0,1,2,3,4),
           #use_folds=(0,),
            checkpoint_name='checkpoint_final.pth',
        )
        
        print('Model_1 input files', os.listdir(input_folder))

        predictor.predict_from_files(input_folder,
                                    model1_output,
                                    save_probabilities=False, overwrite=True,
                                    num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)        
        
        
        
        # cmd = [
        #         'nnUNetv2_predict',
        #         '-d', '99',
        #         '-i', input_folder,
        #         '-o', model1_output,
        #         '-tr', trainer,
        #         '-c', '3d_lowres',
        #         '-p', 'nnUNetPlans',
        #         '--save_probabilities'
        #         #'-device','cpu'
        # ]

        # cmd.append('-f')
        # cmd.extend(folds.split(','))

        # subprocess.run(cmd)
        
        print('Model_1 output', os.listdir(model1_output))
        
        result = SimpleITK.ReadImage(str(self.nnunet_model1_out / "test_1.nrrd"))
        
        outputs = [result,
                   trimesh.primitives.Box(),  #optional visualization task, leave Box() as placeholder if you do not participate
                   trimesh.primitives.Box()]  #optional volumetric meshing task, leave Box() as placeholder if you do not participate
        return outputs



if __name__ == "__main__":
    Segaalgorithm().process()
