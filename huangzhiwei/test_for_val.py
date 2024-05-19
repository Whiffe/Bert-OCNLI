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
from datasets.ocnli_dataset_new import OcnliDataset


# 数据集加载部分
from loaders.mlm_loader_for_val import MLMDataLoader


# 分类器选择部分
from models.bert_classifier import BertClassifier
# from models.bert_classifier_LRandConcat import BertClassifier
# from models.bert_classifier_LSTM_4cell import BertClassifier
# from models.bert_classifier_LSTM_1cell import BertClassifier
# from models.bert_classifier_LSTM_allcell import BertClassifier
# from models.bert_classifier_BLSTM_4cell import BertClassifier
# from models.bert_classifier_BaseLine import BertClassifier
# from models.bert_classifier_GRU_4cell import BertClassifier
# from models.bert_classifier_for_maxpool import BertClassifier
# from models.bert_classifier_4cellmaxpooling import BertClassifier
# from models.bert_classifier_4cellavgpooling import BertClassifier
# from models.bert_classifier_for_avgpooling import BertClassifier

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='./pretrained/chinese-bert-wwm-ext')
    # parser.add_argument('--model_name', type=str, default='./pretrained/chinese-roberta-wwm-ext')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--freeze_pooler', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--seed', type=int, default=1145141919)
    parser.add_argument('--data_path', type=str, default='./data/ocnli_new/ocnli_val.txt')
    parser.add_argument('--result_dir', type=str, default='results')
    

    return parser.parse_args()


def test(configs):

    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(configs.result_dir, exist_ok=True)
    result_path = os.path.join(configs.result_dir, f'{configs.name}.csv')
    
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

    labels = dataset.get_labels()
    results = [{
        'class_id': i,
        'class_name': labels[i],
        'num_samples': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'precision': 0.,
        'recall': 0.,
        'f1': 0.
    } for i in range(configs.num_classes)]

    total_samples = 0
    total_corrects = 0

    with torch.no_grad():
        with tqdm(
            dataloader,
            total=len(dataloader),
            desc='Testing',
            ncols=100
        ) as pbar:
            for input_ids, attention_mask, token_type_ids, labels in pbar:
                outputs = model(input_ids, attention_mask, token_type_ids)
                # Batch size is 1
                outputs = torch.argmax(outputs, dim=-1).flatten().detach().cpu().numpy()
                labels = labels.flatten().detach().cpu().numpy()

                for output, label in zip(outputs, labels):
                    total_samples += 1
                    total_corrects += int(output == label)
                    results[label]['num_samples'] += 1
                    results[label]['tp'] += int(output == label)
                    results[label]['fn'] += int(output != label)
                    results[output]['fp'] += int(output != label)

    for r in results:
        if r['num_samples'] > 0:
            r['precision'] = r['tp'] / (r['tp'] + r['fp']) \
                if r['tp'] + r['fp'] > 0 else 0
            
            r['recall'] = r['tp'] / (r['tp'] + r['fn']) \
                if r['tp'] + r['fn'] > 0 else 0
            
            r['f1'] = 2 * r['precision'] * r['recall'] / (r['precision'] + r['recall']) \
                if r['precision'] + r['recall'] > 0 else 0

    total_accuracy = total_corrects / total_samples
    results.append({
        'class_id': -1,
        'class_name': 'Total',
        'num_samples': total_samples,
        'tp': total_corrects,
        'fp': 0,
        'fn': 0,
        'precision': 'N/A',
        'recall': 'N/A',
        'f1': 'N/A',
        'accuracy': total_accuracy
    })

    df = pd.DataFrame(results)
    df.to_csv(result_path, index=False)


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




