from transformers import MT5Tokenizer, MT5ForConditionalGeneration



import os
import torch
import json
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

# 读ocnli，data_dir是数据集的路径
def read_ocnli(data_dir, trainval_name, isTest=False):
    # 将ocnli解析为前提、假设、标签
    # label_map是标签映射，0、1、2代表三类，3代表无法分类（或者应该去除的数据）。
    label_map = {'entailment':0, 'neutral':1, 'contradiction':2, '-': 3}
    file_name = os.path.join(data_dir, trainval_name)
    
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        if not isTest:
            for row in rows:
                row =  json.loads(row)
                # 去除无法分类的标签
                if row['label'] == '-':
                    continue
                data.append(
                    (
                        (row['sentence1'], row['sentence2']),
                        label_map[row['label']]
                    )
                )
        else:
            for row in rows:
                row =  json.loads(row)
                data.append(
                    (
                        (row['sentence1'], row['sentence2']),
                    )
                )
    return data

class OCNLI_Dataset(Dataset):
    def __init__(
        self,
        dataset,
        device,
        max_length=128,
        pretrain_model_name='t5-base'
    ):
        self.tokenizer = MT5Tokenizer.from_pretrained(pretrain_model_name)
        sentences = [i[0] for i in dataset]
        labels = [i[1] for i in dataset]
        self.texts = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[
                (sentence[0], sentence[1]) for sentence in sentences
            ],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_length=True
        )
        self.input_ids = self.texts['input_ids'].to(device)
        self.attention_mask = self.texts['attention_mask'].to(device)
        '''
        当使用T5模型时，我们不需要包含token_type_ids

        以下是两者之间的主要区别：

        1，BERT：使用token_type_ids（也称为"segment ids"）来区分输入的两个句子。
        例如，对于句子对任务，第一个句子的所有token类型ID设置为0，第二个句子的所有token类型ID设置为1。

        2,T5：使用的是文本到文本的转换任务，不需要token_type_ids。
        它将所有输入视为一个统一的序列，并通过自身的架构进行处理。
        '''
        # 
        

        self.labels = torch.LongTensor(labels).to(device)

    def __len__(self):
        # return len(self.labels)
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]



class OCNLI_Dataset_test(Dataset):
    def __init__(
        self,
        dataset,
        device,
        max_length=128,
        pretrain_model_name='t5-base'
    ):
        self.tokenizer = MT5Tokenizer.from_pretrained(pretrain_model_name)
        sentences = [i[0] for i in dataset]
        self.texts = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[
                (sentence[0], sentence[1]) for sentence in sentences
            ],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_length=True
        )
        self.input_ids = self.texts['input_ids'].to(device)
        self.attention_mask = self.texts['attention_mask'].to(device)

    def __len__(self):
        # return len(self.labels)
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=3, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)  # (输入维度，输出维度)

    def forward(self, features, **kwargs):
        # print("features.shape：",features.shape)
        x = features[:, 0, :]  # features[-1]是一个三维张量，其维度为[批次大小, 序列长度, 隐藏大小]。
        # print("x.shape:",x.shape)
        x = self.dropout(x)  # 这是一种正则化技术，用于防止模型过拟合。在训练过程中，它通过随机将输入张量中的一部分元素设置为0，来增加模型的泛化能力。
        x = self.dense(x)  # 这是一个全连接层，它将输入特征映射到一个新的特征空间。这是通过学习一个权重矩阵和一个偏置向量，并使用它们对输入特征进行线性变换来实现的，方便后续可以引入非线性变换。
        x = torch.tanh(x)  # 这是一个激活函数，它将线性层的输出转换为非线性，使得模型可以学习并表示更复杂的模式。
        x = self.dropout(x)  # 增加模型的泛化能力。
        x = self.out_proj(x)  # 这是最后的全连接层，它将特征映射到最终的输出空间。在这个例子中，输出空间的维度等于分类任务的类别数量。
        # print("x.shape:",x.shape)
        
        return x

class T5Classifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, dropout=0.1):
        super().__init__()
        self.t5 = MT5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.classifier = ClassificationHead(
            hidden_size=self.t5.config.d_model,
            num_classes=num_classes,
            dropout=dropout
        )
    def forward(self, input_ids, attention_mask):
        # 调用T5模型的前向传播
        outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用encoder的最后隐藏状态
        encoder_hidden_state = outputs.last_hidden_state
        
        # 使用encoder的第一个token的表示
        # logits = self.classifier(encoder_hidden_state[:, 0, :])
        logits = self.classifier(encoder_hidden_state)
        
        return logits

def train(model, train_dataset, val_dataset, configs, device, wandb):
    

    train_data = OCNLI_Dataset(train_dataset, device, configs.max_length, configs.pretrain_model_name)
    val_data = OCNLI_Dataset(val_dataset, device, configs.max_length, configs.pretrain_model_name)

    train_dataloader = DataLoader(train_data, batch_size=configs.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=configs.batch_size, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), configs.lr)
    
    
    os.makedirs(configs.checkpoint_dir, exist_ok=True)

    # float('inf') 表示正无穷大
    best_val_loss = float('inf')

    # 设置模型为训练模式
    model.train()
    # 开始进入训练循环
    for epoch in range(configs.epochs):
        # 进度条函数tqdm
        for input_ids, attention_mask, labels in tqdm(train_dataloader):
            optimizer.zero_grad() # 清除上一次迭代的梯度
            output = model(input_ids, attention_mask) # 通过模型得到输出
            loss = criterion(output, labels) # 计算损失
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 使用计算出的梯度更新模型的参数

            output = output.argmax(dim=1).float()
            accuracy = (output == labels).float().mean()
            wandb.log({
                'train loss': loss.item(),
                'train accuracy': accuracy.item()
            })
        print('train loss.item():',loss.item())
        print('train accuracy.item():',accuracy.item())
        state_dict = model.state_dict()  # 获取模型的当前参数状态字典
        checkpoint_path = os.path.join(configs.checkpoint_dir, f'epoch_{epoch + 1}.pt')
        
        try:
            torch.save(state_dict, checkpoint_path)
            print("状态字典已成功保存到文件:", checkpoint_path)
        except Exception as e:
            print("状态字典保存失败！")
            print("错误信息：", str(e))

        # wandb.save(checkpoint_path)

        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        val_loss = 0.0
        val_corrects = 0

        # 设置模型为评估模式
        model.eval()
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for input_ids, attention_mask, labels in val_dataloader:
                output = model(input_ids, attention_mask)
                output = output.argmax(dim=1).float()
                labels = labels.float()

                loss = criterion(output, labels)
                # loss.item() 是当前批次的平均损失
                # input_ids.size(0) 是当前批次的样本数量
                # 通过乘以 input_ids.size(0)，原代码将平均损失转换为总损失，累积到 val_loss 中
                val_loss += loss.item() * input_ids.size(0)
                val_corrects += (output == labels).sum().item()
        print("val_corrects:",val_corrects)
        print("len(val_dataset):",len(val_dataset))
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects / len(val_dataset)
        wandb.log({
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(configs.checkpoint_dir, 'best_model.pt'))
            print(f"best epoch: {epoch}")

        print('val_loss:',val_loss)
        print('val_acc:',val_acc)
        model.train()

def test(model, test_dataset, configs, device):
    test_data = OCNLI_Dataset_test(test_dataset, device, configs.max_length, configs.pretrain_model_name)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # state_dict = torch.load(os.path.join(configs.checkpoint_dir, f'epoch_{configs.epochs}.pt'))
    state_dict = torch.load(os.path.join(configs.checkpoint_dir, f'best_model.pt'))

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        test_preds = []
        
        for input_ids, attention_mask in test_dataloader:
            output = model(input_ids, attention_mask)
            output = output.argmax(dim=-1).float()
            # test_preds.append(output)
            test_preds.append(output.cpu().numpy())  # 将 Tensor 移动到 CPU 并转换为 numpy 数组
    
    print("len(test_preds):",len(test_preds))
    # 将预测结果展平为一维列表
    test_preds = [pred.item() for preds in test_preds for pred in preds]

    test_df = pd.read_json(os.path.join(configs.data_dir, configs.test_name), lines=True)

    test_df["label"] = test_preds
    test_df["label"] = test_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
    test_df = test_df[["label", "id"]]
    test_df.to_json(os.path.join(configs.checkpoint_dir, f"ocnli_50k_predict.json"), orient="records", lines=True, force_ascii=False)

    # os.system(f"zip -j {configs.checkpoint_dir}/ocnli_50k_predict.zip {configs.checkpoint_dir}/ocnli_50k_predict.json")
