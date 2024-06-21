from DiffuAug.srcs.classification.train.train import train
from DiffuAug.srcs.classification.metrics.roc_curve import *
from DiffuAug.srcs import utility

def main():
    YAML_PATH = r"/workspace/DiffuAug/exp_settings/configs/classification/normal_test_settings/balanced/resnet18_total_200.yaml"
    OPTION = "train"
    
    utility.set_seed()
    cfg = utility.load_config(YAML_PATH)    
    cfg = utility.dict2namespace(cfg)
    
    if OPTION == "train":
        train(cfg)
        
    elif OPTION == "test":
        pred_result_csv_path = r"/data/results/classification/predicted_csv/resnet18/aug_test/balacned_300/predicted_30.csv"
        save_curve_png_path = r"/data/results/classification/plot"
    
        draw_roc_curve(pred_result_csv_path, save_curve_png_path)
        compute_auc_with_slices(pred_result_csv_path)
        compute_acc_with_slices(pred_result_csv_path)


if __name__ == '__main__':
    main()