from DiffuAug.srcs import utility

def main():
    YAML_PATH = r"/workspace/DiffuAug/exp_settings/configs/classification/resnet18.yaml"
    
    utility.set_seed()
    cfg = utility.load_config(YAML_PATH)
    print("cfg: ", cfg)
    
    # cfg = utility.dict2namespace(cfg)
    
    # train(cfg)


if __name__ == '__main__':
    main()