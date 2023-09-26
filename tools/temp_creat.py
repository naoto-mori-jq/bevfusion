import mmcv
import os.path as osp

# Define the dataset root
dataset_root = "../../dataset/nuscenes/"

# Paths to the input files
train_file_path = osp.join(dataset_root, "nuscenes_infos_train.pkl")
val_file_path = osp.join(dataset_root, "nuscenes_infos_val.pkl")

# Load the data from the input files
train_nusc_infos = mmcv.load(train_file_path)
val_nusc_infos = mmcv.load(val_file_path)

# Combine the data
data = train_nusc_infos
data['infos'] = data['infos'] + val_nusc_infos['infos']

# Path to the output file
info_all_path = osp.join(dataset_root, "nuscenes_infos_all.pkl")

# Dump the combined data to the output file
mmcv.dump(data, info_all_path)

# Output the path to the created file for confirmation
print(f"Combined data saved to {info_all_path}")
