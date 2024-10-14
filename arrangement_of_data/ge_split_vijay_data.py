import os
import json

def get_gt_files(folder_path):
    gt_files = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return gt_files

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file name contains '_gt'
        if '_gt' in filename:
            # Remove the '.nii.gz' extension if present
            name_without_extension = filename.replace('.nii.gz', '')
            
            # Append the file name (without extension) to the list
            print(name_without_extension[:-3])
            gt_files.append(name_without_extension[:-3])

    return gt_files

data = []


f1_dict = {'train': [], 'val': []}
train_f1 = get_gt_files(r'C:\My_Data\Vijay\F1\train')
val_f1 = get_gt_files(r'C:\My_Data\Vijay\F1\val')
f1_dict['train'] = train_f1
f1_dict['val'] = val_f1
data.append(f1_dict)


f2_dict = {'train': [], 'val': []}
train_f2 = get_gt_files(r'C:\My_Data\Vijay\F2\train')
val_f2 = get_gt_files(r'C:\My_Data\Vijay\F2\val')
f2_dict['train'] = train_f2
f2_dict['val'] = val_f2
data.append(f2_dict)


f3_dict = {'train': [], 'val': []}
train_f3 = get_gt_files(r'C:\My_Data\Vijay\F3\train')
val_f3 = get_gt_files(r'C:\My_Data\Vijay\F3\val')
f3_dict['train'] = train_f3
f3_dict['val'] = val_f3
data.append(f3_dict)


f4_dict = {'train': [], 'val': []}
train_f4 = get_gt_files(r'C:\My_Data\Vijay\F4\train')
val_f4 = get_gt_files(r'C:\My_Data\Vijay\F4\val')
f4_dict['train'] = train_f4
f4_dict['val'] = val_f4
data.append(f4_dict)


f5_dict = {'train': [], 'val': []}
train_f5 = get_gt_files(r'C:\My_Data\Vijay\F5\train')
val_f5 = get_gt_files(r'C:\My_Data\Vijay\F5\val')
f5_dict['train'] = train_f5
f5_dict['val'] = val_f5
data.append(f5_dict)

# Save the dictionary as a JSON file
def save_splits_json(splits_dict, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(splits_dict, json_file, indent=4)

# Usage
file_path = r'C:\My_Data\Vijay/'+ 'splits_final.json'
save_splits_json(data, file_path)









