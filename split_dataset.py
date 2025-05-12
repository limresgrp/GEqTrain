import os
import random
import shutil

def split_dataset(source_folder, training_folder, validation_folder, test_folder, num_training, num_validation):
    # Ensure destination folders exist
    os.makedirs(training_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all .npz files in the source folder
    all_files = [f for f in os.listdir(source_folder) if f.endswith('.npz')]

    # Shuffle the files randomly
    random.shuffle(all_files)

    # Split the files into training, validation, and test sets
    training_files = all_files[:num_training]
    validation_files = all_files[num_training:num_training + num_validation]
    test_files = all_files[num_training + num_validation:]

    # Move files to their respective folders
    for file in training_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(training_folder, file))
    for file in validation_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(validation_folder, file))
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))

    print(f"Moved {len(training_files)} files to {training_folder}")
    print(f"Moved {len(validation_files)} files to {validation_folder}")
    print(f"Moved {len(test_files)} files to {test_folder}")

# Example usage
source_folder = "/capstor/scratch/cscs/dangiole/qm9/npz/test"
training_folder = "/capstor/scratch/cscs/dangiole/qm9/npz/train"
validation_folder = "/capstor/scratch/cscs/dangiole/qm9/npz/valid"
test_folder = "/capstor/scratch/cscs/dangiole/qm9/npz/test"
num_training = 10000
num_validation = 0

split_dataset(source_folder, training_folder, validation_folder, test_folder, num_training, num_validation)