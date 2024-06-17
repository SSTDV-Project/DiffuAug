import os

from DiffuAug.srcs.dataset_processing.metadata_processing.metadata_processing import (make_metadata, split_train_test_val_csv, test_leak_data)

def main():    
    OPTION="split_data"
    
    if OPTION == "make_metadata":
        data_dir = r"/data/duke_data/size_64/split_datalabel"
        output_csv_path = r"/workspace/DiffuAug/metadata/classification/csv"
        
        make_metadata(
            data_dir=data_dir,
            output_csv_path=output_csv_path,
            output_csv_name="duke_data_total_200_balanced.csv",
            option="balanced",
            class_num_data=100
        )
        
    elif OPTION == "split_data":
        duke_csv_path = r"/workspace/DiffuAug/metadata/classification/csv/duke_data_total_200_balanced.csv"
        csv_output_root_path = r"/workspace/DiffuAug/metadata/classification/csv/0.8_0.1_0.1_balanced_200"
        split_train_test_val_csv(
            meta_csv_path=duke_csv_path,
            csv_output_root_path=csv_output_root_path
        )
        
    elif OPTION == "leak_data":
        splitted_csv_root_path = r"/workspace/DiffuAug/metadata/classification/csv/0.8_0.1_0.1"
        train_data_path = os.path.join(splitted_csv_root_path, 'train_dataset.csv')
        val_data_path = os.path.join(splitted_csv_root_path, 'val_dataset.csv')
        test_data_path = os.path.join(splitted_csv_root_path, 'test_dataset.csv')
        
        test_leak_data(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path
        )
        

if __name__ == "__main__":
    main()