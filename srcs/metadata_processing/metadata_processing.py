import os

import pandas as pd

def make_metadata(data_dir, output_csv_path):
    print("Making metadata...")
    
    output_csv_path = os.path.join(output_csv_path, "duke_metadata.csv")
    colums = ['data_path', 'label']
    labeld_df = pd.DataFrame(columns=colums)
    
    for idx, label in enumerate(['neg', 'pos']):
        data_path = os.path.join(data_dir, label)
        for f in os.listdir(data_path):
            df = pd.DataFrame(columns=colums)
            file_name = os.path.join(data_path, f)
            df['data_path'] = [file_name]
            df['label'] = [label]
            labeld_df = pd.concat([labeld_df, df], ignore_index=True)
    
    labeld_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print("Label neg length:", len(labeld_df[labeld_df['label'] == 'neg']))
    print("Label pos length:", len(labeld_df[labeld_df['label'] == 'pos']))
    print("Total length:", len(labeld_df))
    print("Metadata is saved at", output_csv_path)
    