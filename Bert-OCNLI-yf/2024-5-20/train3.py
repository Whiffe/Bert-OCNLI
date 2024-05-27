'''
python train3.py --epochs 1 --train_name train.3k.json
python train3.py --epochs 10 --train_name train.50k.json

'''
import torch
import argparse
import numpy as np
from bert3 import read_ocnli, BertClassifier, train, test
import wandb


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./ocnli', type=str)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--train_name', default='train.50k.json', type=str)
    parser.add_argument('--val_name', default='dev.json', type=str)
    parser.add_argument('--test_name', default='test.json', type=str)
    parser.add_argument('--pretrain_model_name', default='bert-base-chinese', type=str)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint_dir', default='./checkpoint', type=str)


    return parser.parse_args()

if __name__ == '__main__':
    configs = argparser()   # 加载参数配置

    # 配置 wandb 运行
    wandb.init(
        project="BERT-OCNLI-YF",   # 项目名称
        name="bert-base-epcoh10-lr5e-5",    # 运行名称
        # mode="online", # 运行模式
        mode="disabled", # 运行模式
        entity="feildingyf"
    )

    '''
    wandb.config.pretrain_model_name = configs.pretrain_model_name
    wandb.config.num_classes = configs.num_classes
    wandb.config.dropout = configs.dropout
    wandb.config.batch_size = configs.batch_size
    wandb.config.max_length = configs.max_length
    wandb.config.lr = configs.lr
    wandb.config.epochs = configs.epochs
    '''
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

    model = BertClassifier(
        pretrained_model_name='bert-base-chinese',
        num_classes=configs.num_classes,
        dropout=configs.dropout
        ).to(device)
    
    wandb.watch(model, log='all')

    train(model, train_dataset, val_dataset, configs, device, wandb)
    test(model, test_dataset, configs, device)

    wandb.finish()
