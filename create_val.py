import os
import shutil
import random
import argparse



def move_files(folder1, folder2, N_val):
    # Get the list of files in folder1
    files = [f for f in os.listdir(folder1) if f.startswith("mol_") and f.endswith(".npz")]

    if len(files) < N_val:
        raise ValueError(f"Not enough files in {folder1} to move {N_val} files")

    # Randomly select N_val files, without replacement
    selected_files = random.sample(files, N_val)

    # Move each selected file to folder2
    for file in selected_files:
        shutil.move(os.path.join(folder1, file), os.path.join(folder2, file))


def copy_files(folder1, folder2, N_val):
    # Get the list of files in folder1
    files = [f for f in os.listdir(folder1) if f.startswith("mol_") and f.endswith(".npz")]

    if len(files) < N_val:
        raise ValueError(f"Not enough files in {folder1} to move {N_val} files")

    # Randomly select N_val files
    selected_files = random.sample(files, N_val)

    # Copy each selected file to folder2
    for file in selected_files:
        shutil.copy(os.path.join(folder1, file), os.path.join(folder2, file))


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(description="Move random files from folder1 to folder2")
    parser.add_argument('folder1', type=str, help="Source folder")
    parser.add_argument('folder2', type=str, help="Destination folder")
    parser.add_argument('N_val', type=int, help="Number of files to move")

    args = parser.parse_args()

    move_files(args.folder1, args.folder2, args.N_val)

# https://www.tensorflow.org/datasets/catalog/qm9
# python create_val.py /storage_common/nobilm/qm_npz/train /storage_common/nobilm/qm_npz/validation 5
# python create_val.py /storage_common/nobilm/qm_npz/train /storage_common/nobilm/qm_npz/validation 17743
# python create_val.py /storage_common/nobilm/qm_npz/train /storage_common/nobilm/qm_npz/test 13083


