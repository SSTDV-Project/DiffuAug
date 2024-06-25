import os

from DiffuAug.srcs.dataset_processing.metadata_processing.metadata_processing import *
from DiffuAug.srcs.dataset_processing.metadata_processing.metadata_patient_processing import *

def main():
    SLICE_OPTION=""
    PATIENT_OPTION = "patient_gather"
    
    # Slice-wise metadata processing
    if SLICE_OPTION == "make_metadata":
        data_dir = r"/data/duke_data/size_64/split_datalabel"
        output_csv_path = r"/workspace/DiffuAug/metadata/classification/csv"
        
        make_metadata(
            data_dir=data_dir,
            output_csv_path=output_csv_path,
            output_csv_name="duke_data_total_200_balanced.csv",
            option="balanced",
            class_num_data=100
        )
        
    elif SLICE_OPTION == "split_data":
        duke_csv_path = r"/workspace/DiffuAug/metadata/classification/csv/duke_data_total_200_balanced.csv"
        csv_output_root_path = r"/workspace/DiffuAug/metadata/classification/csv/0.8_0.1_0.1_balanced_200"
        split_train_test_val_csv(
            meta_csv_path=duke_csv_path,
            csv_output_root_path=csv_output_root_path
        )
        
    elif SLICE_OPTION == "origin_plus_augdata":
        origin_train_csv_path = r"/workspace/DiffuAug/metadata/classification/csv/0.8_0.1_0.1_balanced_200/train_dataset.csv"
        aug_file_parent_path = r"/data/results/generation/sampling/cfg/imbalanced/sampling_imgs/ddim/epoch_70/p_uncond_0.2/w_4.0"
        output_csv_path = r"/workspace/DiffuAug/metadata/classification/aug_csv/0.8_0.1_0.1_balanced_200+50"
        
        origin_plus_augdata(
            origin_train_dataset_path=origin_train_csv_path,
            aug_file_parent_path=aug_file_parent_path,
            output_csv_path=output_csv_path
        )
        
    elif SLICE_OPTION == "test_leak_data":
        splitted_csv_root_path = r"/workspace/DiffuAug/metadata/classification/csv/0.8_0.1_0.1"
        train_data_path = os.path.join(splitted_csv_root_path, 'train_dataset.csv')
        val_data_path = os.path.join(splitted_csv_root_path, 'val_dataset.csv')
        test_data_path = os.path.join(splitted_csv_root_path, 'test_dataset.csv')
        
        test_leak_data(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path
        )
        
    # Patient-wise metadata processing
    if PATIENT_OPTION == "patient_gather":
        data_dir = r"/data/duke_data/patients/png_out_64"
        save_csv_path = r"/workspace/DiffuAug/metadata/patient_unit/patient100"
        gather_patient_data(data_dir, save_csv_path)
    

if __name__ == "__main__":
    main()