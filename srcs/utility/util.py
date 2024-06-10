import os
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import yaml


def load_config(yaml_path):
    """
    yaml 파일을 읽어서 dictionary 형태로 반환합니다.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def save_model(net, states, save_path):
    """
    학습한 모델의 가중치를 저장합니다. 
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # GPU를 병렬로 사용하는 경우, 모델을 저장할 때 module을 붙여주어 저장.
    if isinstance(net, nn.DataParallel) or isinstance(net, nn.parallel.DistributedDataParallel):
        states['model_state'] = net.module.state_dict()
        torch.save(states, save_path)
    else:    
        torch.save(states, save_path)


def dict2namespace(config):
    """ 
    dictionary를 namespace로 변환 
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    
    return namespace


def set_seed(seed=990912):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(cfg):
    log_path = os.path.join(cfg.paths.exp_path, cfg.paths.log_dir)
    
    log_folder = Path(log_path)
    log_folder.mkdir(parents=True, exist_ok=True)
    
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_path, "stdout.txt"))
    
    formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(cfg.log.verbose.upper())