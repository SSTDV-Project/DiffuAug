from DiffuAug.srcs.classification.train.train import train
from DiffuAug.srcs.classification.metrics.roc_curve import *
from DiffuAug.srcs import utility

def main():
    YAML_PATH = r"/workspace/DiffuAug/exp_settings/configs/classification/aug_test_setttings/patient/ddim/p_uncond_0.2/resnet18_total_p100+aug5200_diff_w.yaml"
    OPTION = "test"
    
    utility.set_seed()
    cfg = utility.load_config(YAML_PATH)    
    cfg = utility.dict2namespace(cfg)
    
    if OPTION == "train":
        train(cfg)
        
        # w_list = [
        #     0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        #     1.0, 2.0, 3.0, 4.0
        #     ]
        
        # for w in w_list:
        #     cfg.paths.train_csv_path = f"/workspace/DiffuAug/metadata/classification/patient_csv/patient100/aug_csv/ddim/p_uncond_0.2/aug_train_ddim_w_{w}.csv"
        #     cfg.paths.exp_path = f"/data/results/classification/exps/aug/patients/ddim/p_uncond_0.2/differ_w/w_{w}"
        #     print("train_csv_path: ", cfg.paths.train_csv_path)
        #     print("exp_path: ", cfg.paths.exp_path)
            
        #     train(cfg)
        
    elif OPTION == "test":
        pred_result_csv_path = r"/data/results/classification/exps/aug/patients/ddim/p_uncond_0.2/differ_w/w_4.0/predict_result/predicted_best_7.csv"
        # save_curve_png_path = r"/data/results/classification/exps/aug/slices/balanced/o200+aug50+aug50/plot"
    
        # draw_roc_curve(pred_result_csv_path, save_curve_png_path)
        sensitivity, specificity, roc_auc = compute_auc_with_slices(pred_result_csv_path)
        acc = compute_acc_with_slices(pred_result_csv_path)
        
        print(f"AUC: {roc_auc:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}\n")
        print(f"Accuracy: {acc:.2f}\n")


if __name__ == '__main__':
    main()