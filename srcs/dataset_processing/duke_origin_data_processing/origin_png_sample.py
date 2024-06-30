import os
import random
import shutil

def copy_random_png_files(src_dir, dest_dir, num_files=2600):
    # Ensure the source directory exists
    if not os.path.isdir(src_dir):
        raise ValueError(f"The source directory {src_dir} does not exist.")
    
    # Ensure the destination directory exists or create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all PNG files in the source directory
    all_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    
    # Check if there are enough files to copy
    if len(all_files) < num_files:
        raise ValueError(f"Not enough PNG files in the source directory. Found {len(all_files)} files.")
    
    # Randomly sample the files
    random_files = random.sample(all_files, num_files)
    
    # Copy the selected files to the destination directory
    for file_name in random_files:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy(src_file, dest_file)
    
    print(f"Copied {num_files} PNG files to {dest_dir}")


def main():
    # Example usage
    src_directory = "/data/duke_data/size_64/split_datalabel/pos"
    dest_directory = "/data/duke_data/size_64/random_balanced_origin_data/pos"
    copy_random_png_files(src_directory, dest_directory)


if __name__ == '__main__':
    main()