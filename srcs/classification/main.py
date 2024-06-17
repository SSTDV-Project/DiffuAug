from DiffuAug.srcs.classification.train.train import train
from DiffuAug.srcs import utility

def main():
    YAML_PATH = r"/workspace/DiffuAug/exp_settings/configs/classification/resnet18_total_200.yaml"
    OPTION = "train"
    
    utility.set_seed()
    cfg = utility.load_config(YAML_PATH)    
    cfg = utility.dict2namespace(cfg)
    
    if OPTION == "train":
        train(cfg)


if __name__ == '__main__':
    main()