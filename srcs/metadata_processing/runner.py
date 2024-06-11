import os

from DiffuAug.srcs.metadata_processing.metadata_processing import make_metadata

def main(OPTION):
    if OPTION == "make_metadata":
        data_dir = r"/data/duke_data/size_64/split_datalabel"
        output_csv_path = r"/workspace/DiffuAug/metadata/classification/csv"
        
        make_metadata(
            data_dir=data_dir,
            output_csv_path=output_csv_path
        )


if __name__ == "__main__":
    main(OPTION="make_metadata")