''' 
python main.py --pretrain_model_name google-t5/t5-base --epochs 1
python main.py --pretrain_model_name google/mt5-base --epochs 1
'''
import torch
import argparse
import numpy as np
from model import read_ocnli, T5Classifier, train, test
import wandb
import random

# 为 Python 的随机数生成器、NumPy、PyTorch（包括其 CUDA 后端）的随机数生成器设置相同的随机种子
# 以确保在多次运行代码时，生成的随机数序列保持一致。

random_seed = 1145141919
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./ocnli', type=str)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--train_name', default='train.50k.json', type=str)
    parser.add_argument('--val_name', default='dev.json', type=str)
    parser.add_argument('--test_name', default='test.json', type=str)
    parser.add_argument('--pretrain_model_name', default='bert-base-chinese', type=str)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)


    return parser.parse_args()

if __name__ == '__main__':

    
    configs = argparser()   # 加载参数配置

    # 配置 wandb 运行
    wandb.init(
        project="ROBERTA-OCNLI-yf",   # 项目名称
        name="roberta",    # 运行名称
        # mode="online", # 运行模式
        mode="disabled", # 运行模式
        entity="feildingyf"
    )
    wandb.config.update({
        'pretrain_model_name': configs.pretrain_model_name,
        'num_classes': configs.num_classes,
        'dropout': configs.dropout,
        'batch_size': configs.batch_size,
        'max_length': configs.max_length,
        'lr': configs.lr,
        'epochs': configs.epochs
    })

    train_dataset = read_ocnli(configs.data_dir, configs.train_name)
    
    val_dataset = read_ocnli(configs.data_dir, configs.val_name)
    test_dataset = read_ocnli(configs.data_dir, configs.test_name, isTest=True)

    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = T5Classifier(
        pretrained_model_name=configs.pretrain_model_name,
        num_classes=configs.num_classes,
        dropout=configs.dropout
        ).to(device)
    
    wandb.watch(model, log='all')

    train(model, train_dataset, val_dataset, configs, device, wandb)
    test(model, test_dataset, configs, device)

    wandb.finish()
