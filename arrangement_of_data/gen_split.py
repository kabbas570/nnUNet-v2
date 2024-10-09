import json

# Function to generate fold splits
def generate_folds(total_instances=360, fold_size=72):
    data = []
    
    for fold in range(5):  # Generate 5 folds
        fold_dict = {'train': [], 'val': []}
        
        val_start = fold * fold_size
        val_end = val_start + fold_size
        
        # Validation split (72 instances per fold)
        val_instances = [str(i).zfill(3) + '_LA_ES' for i in range(val_start + 1, val_end + 1)]
        val_instances += [str(i).zfill(3) + '_LA_ED' for i in range(val_start + 1, val_end + 1)]
        
        # Training split (all others except validation instances)
        train_instances = [str(i).zfill(3) + '_LA_ES' for i in range(1, total_instances + 1) if i <= val_start or i >= val_end + 1]
        train_instances += [str(i).zfill(3) + '_LA_ED' for i in range(1, total_instances + 1) if i <= val_start or i >= val_end + 1]
        
        fold_dict['train'] = train_instances
        fold_dict['val'] = val_instances
        
        data.append(fold_dict)
    
    return data

# Generate the folds
folds_data = generate_folds()

# Save the dictionary as a JSON file
def save_splits_json(splits_dict, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(splits_dict, json_file, indent=4)

# Usage
file_path = r'C:\Users\Abbas Khan\Downloads\Nay/' + 'all_folds.json'
save_splits_json(folds_data, file_path)

print(f"All five folds saved successfully at {file_path}")


import json

def load_data_split(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data_split = json.load(file)
    
    return data_split

# Usage
file_path = r"C:\Users\Abbas Khan\Downloads\Nay\splits_final.json"  # Provide the correct file path
data_split = load_data_split(file_path)

file_path = r"C:\Users\Abbas Khan\Downloads\Nay\all_folds.json"  # Provide the correct file path
my_split = load_data_split(file_path)

# Print the contents of the data split
print(json.dumps(data_split, indent=4))

f1 = data_split[0]
f2 = data_split[1]
f3 = data_split[2]
f4 = data_split[3]
f5 = data_split[4]

data = []

f1_dict = {'train': [], 'val': []}

train_f1 = []
for i in range(1,73):
    
    train_f1.append(str(i).zfill(3) + '_LA_ES')
    train_f1.append(str(i).zfill(3) + '_LA_ED')
    
val_f1 = []
for i in range(73,361):
    
    val_f1.append(str(i).zfill(3) + '_LA_ES')
    val_f1.append(str(i).zfill(3) + '_LA_ED')
    

f1_dict['train'] = train_f1
f1_dict['val'] = val_f1

data.append(f1_dict)

f2_dict = {'train': [], 'val': []}

train_f2 = []
for i in range(1,289):
    
    train_f2.append(str(i).zfill(3) + '_LA_ES')
    train_f2.append(str(i).zfill(3) + '_LA_ED')
    
val_f2 = []
for i in range(289,361):
    
    val_f2.append(str(i).zfill(3) + '_LA_ES')
    val_f2.append(str(i).zfill(3) + '_LA_ED')
    

f2_dict['train'] = train_f2
f2_dict['val'] = val_f2
  
data.append(f2_dict)


# Save the dictionary as a JSON file
def save_splits_json(splits_dict, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(splits_dict, json_file, indent=4)

# Usage
file_path = r'C:\Users\Abbas Khan\Downloads\Nay/'+ 'my_folds.json'
save_splits_json(data, file_path)
