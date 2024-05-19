'''
python train.py --model_name ./pretrained/bert-base-chinese --num_classes 3 --dropout 0.1 --batch_size 32 --max_length 128 

--freeze_pooler是啥
'''
# 
import os
import wandb
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW

# 数据集处理部分
# from datasets.toutiao_dataset import ToutiaoDataset
# from datasets.cnews_dataset import CnewsDataset
# from datasets.ocnli_dataset import OcnliDataset
from datasets.ocnli_dataset_new import OcnliDataset

# 数据加载部分
from loaders.mlm_loader_for_val import MLMDataLoader

# 模型分类器部分
from models.bert_classifier import BertClassifier
# from models.bert_classifier_LRandConcat import BertClassifier
# from models.bert_classifier_LSTM_4cell import BertClassifier
# from models.bert_classifier_LSTM_1cell import BertClassifier
# from models.bert_classifier_LSTM_allcell import BertClassifier
# from models.bert_classifier_BLSTM_4cell import BertClassifier
# from models.bert_classifier_GRU_4cell import BertClassifier
# from models.bert_classifier_BaseLine import BertClassifier

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='./pretrained/chinese-bert-wwm-ext')
    # parser.add_argument('--model_name', type=str, default='./pretrained/chinese-roberta-wwm-ext')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--freeze_pooler', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--project', type=str, default='bert_ocnli_new_bert_wwm_classification')
    # parser.add_argument('--project', type=str, default='bert_ocnli_new_roberta_wwm_classification')
    parser.add_argument('--entity', type=str, default='akccc')
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--seed', type=int, default=1145141919)
    parser.add_argument('--data_path', type=str, default='./data/ocnli_new/ocnli_train.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--val_data_path', type=str, default='./data/ocnli_new/ocnli_val.txt')  # 验证集位置
    parser.add_argument('--patience', type=int, default=5)  # 添加一个新的参数，用于设置早停的耐心值

    return parser.parse_args()


def train(configs):

    wandb.init(
        project=configs.project,
        # entity=configs.entity,
        name=configs.exp_name,
        mode='disabled',
    )

    wandb_config = wandb.config
    wandb_config.model_name = configs.model_name
    wandb_config.num_classes = configs.num_classes
    wandb_config.dropout = configs.dropout
    wandb_config.freeze_pooler = configs.freeze_pooler
    wandb_config.batch_size = configs.batch_size
    wandb_config.max_length = configs.max_length
    wandb_config.lr = configs.lr
    wandb_config.epochs = configs.epochs
    wandb_config.device = configs.device
    wandb_config.seed = configs.seed

    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    checkpoint_dir = os.path.join(configs.checkpoint_dir, configs.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataset = OcnliDataset(configs.data_path)
    val_dataset = OcnliDataset(configs.val_data_path)  # 加载验证集


    # bert中实现掩码
    dataloader = MLMDataLoader(
        dataset=dataset,
        batch_size=configs.batch_size,
        max_length=configs.max_length,
        shuffle=True,
        drop_last=True,
        device=configs.device,
        tokenizer_name=configs.model_name
    )

    val_dataloader = MLMDataLoader(
        dataset=val_dataset,
        batch_size=configs.batch_size,
        max_length=configs.max_length,
        shuffle=False,
        drop_last=False,
        device=configs.device,
        tokenizer_name=configs.model_name
    )

    model = BertClassifier(
        # model_name=configs.model_name,
        pretrained_model_name=configs.model_name,
        num_classes=configs.num_classes,
        # dropout=configs.dropout,
        # freeze_pooler=configs.freeze_pooler
    ).to(configs.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.lr
    )

    best_val_loss = float('inf')
    wandb.watch(model, log='all')
    model.train()

    for epoch in range(configs.epochs):
        with tqdm(
            dataloader,
            total=len(dataloader),
            desc=f'Epoch {epoch + 1}/{configs.epochs}',
            unit='batch',
            ncols=100
        ) as pbar:
            for input_ids, attention_mask, token_type_ids, labels in pbar:
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                outputs = outputs.argmax(dim=1)
                accuracy = (outputs == labels).float().mean()

                pbar.set_postfix(
                    loss=f'{loss.item():.3f}',
                    accuracy=f'{accuracy.item():.3f}'
                )

                wandb.log({
                    'loss': loss.item(),
                    'accuracy': accuracy.item()
                })

        # 保存模型训练过程中的参数
        state_dict = model.state_dict() # 获取模型的当前参数状态字典。

        '''
        根据模型的不同部分的冻结状态，选择性地保存参数。
        在模型中，通常有一些层是预训练模型的一部分，如BERT或RoBERTa的权重，
        这些层在微调时可能会被冻结以保持预训练模型的特征提取能力。
        而pooler层通常是针对特定任务添加的额外层，可能在微调时需要更新。
        因此，根据模型的设计和训练需求，可以选择性地保存参数。
        '''
        # 根据模型中是否冻结pooler层来选择保存哪些参数
        # 检查模型中是否冻结了pooler层
        if model.freeze_pooler:
            #  如果pooler层被冻结，`condition` 函数将保留所有不包含 `'bert'` 字符串的参数。
            condition = lambda k: 'bert' not in k
        else:
            # `condition` 函数将保留所有不包含 `'bert'` 字符串或者包含 `'pooler'` 字符串的参数。
            condition = lambda k: 'bert' not in k or 'pooler' in k

        state_dict = {k: v for k, v in state_dict.items() if condition(k)}

        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')
        # print("构建的文件路径为:", checkpoint_path)
        # print(f"周期数为{epoch+1}")
        try:
            torch.save(state_dict, checkpoint_path)
            print("状态字典已成功保存到文件:", checkpoint_path)
        except Exception as e:
            print("状态字典保存失败！")
            print("错误信息：", str(e))
        wandb.save(checkpoint_path)


        # 进行验证集，并实现早停策略
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, labels in val_dataloader:
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                # loss.item() 是当前批次的平均损失
                # input_ids.size(0) 是当前批次的样本数量
                # 通过乘以 input_ids.size(0)，原代码将平均损失转换为总损失，累积到 val_loss 中。这
                val_loss += loss.item() * input_ids.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        
        # Check if we need to save the model and if early stopping is needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(configs.checkpoint_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print('Early stopping triggered.')
                break

        model.train()

    wandb.finish()


if __name__ == '__main__':
    configs = argparser()   # 加载参数配置

    if configs.name is None:
        configs.exp_name = \
            f'{os.path.basename(configs.model_name)}' + \
            f'{"_fp" if configs.freeze_pooler else ""}' + \
            f'_b{configs.batch_size}_e{configs.epochs}' + \
            f'_len{configs.max_length}_lr{configs.lr}'
        
    if configs.device is None:
        configs.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    train(configs)
