import os
import random
import argparse
import json
import zipfile
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

# 数据集处理部分
# from datasets.toutiao_dataset import ToutiaoDataset
# from datasets.cnews_dataset import CnewsDataset
from datasets.ocnli_dataset_new_test import OcnliDataset

# 数据加载部分
from loaders.mlm_loader_for_test import MLMDataLoader

# 分离器部分
from models.bert_classifier import BertClassifier
# from models.bert_classifier_LRandConcat import BertClassifier
# from models.bert_classifier_LSTM_4cell import BertClassifier
# from models.bert_classifier_LSTM_1cell import BertClassifier
# from models.bert_classifier_LSTM_allcell import BertClassifier
# from models.bert_classifier_BLSTM_4cell import BertClassifier
# from models.bert_classifier_for_maxpool import BertClassifier
# from models.bert_classifier_4cellmaxpooling import BertClassifier
# from models.bert_classifier_4cellavgpooling import BertClassifier
# from models.bert_classifier_for_avgpooling import BertClassifier

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='./pretrained/chinese-bert-wwm-ext')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--freeze_pooler', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--seed', type=int, default=1145141919)
    parser.add_argument('--data_path', type=str, default='./data/ocnli_new/ocnli_test.txt')
    parser.add_argument('--result_dir', type=str, default='results')
    

    return parser.parse_args()


def test(configs):

    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    result_path = 'ocnli_50k_predict.json'  # 直接使用文件名作为路径
    
    dataset = OcnliDataset(configs.data_path)

    dataloader = MLMDataLoader(
        dataset, 
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

    state_dict = model.state_dict()
    checkpoint = torch.load(configs.checkpoint_path, map_location=configs.device)
    state_dict.update(checkpoint)
    model.load_state_dict(state_dict)

    model.eval()

    labels_map = dataset.get_labels()  # 获取类别名称的映射

    json_results = []

    with torch.no_grad():
        with tqdm(
            dataloader,
            total=len(dataloader),
            desc='Testing',
            ncols=100
        ) as pbar:
            for i, (input_ids, attention_mask, token_type_ids) in enumerate(pbar):
                outputs = model(input_ids, attention_mask, token_type_ids)
                outputs = torch.argmax(outputs, dim=-1).flatten().detach().cpu().numpy()

                for output in outputs:
                    result = {"label": labels_map[output], "id": i}  # 将类别ID映射回类别名称
                    json_results.append(result)

    # 将结果保存为json文件
    with open(result_path, 'w') as f:
        for result in json_results:
            json.dump(result, f, separators=(',', ':'))  # 设置分隔符为逗号和冒号，没有空格
            f.write('\n')  # 在每个结果后面添加一个换行符


    # 将json文件压缩为zip文件
    with zipfile.ZipFile('ocnli_50k_predict_30.zip', 'w') as zipf:
        zipf.write(result_path)

    # 删除json文件
    os.remove(result_path)

    

if __name__ == '__main__':
    configs = argparser()

    if configs.name is None:
        configs.exp_name = \
            f'{os.path.basename(configs.model_name)}' + \
            f'{"_fp" if configs.freeze_pooler else ""}' + \
            f'_len{configs.max_length}'
        
    if configs.device is None:
        configs.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    test(configs)