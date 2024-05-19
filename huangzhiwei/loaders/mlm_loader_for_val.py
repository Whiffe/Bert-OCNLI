import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class MLMDataLoader:
    def __init__(
        self,
        dataset,
        batch_size=16,
        max_length=128,
        shuffle=True,
        drop_last=True,
        device=None,
        tokenizer_name='chinese-bert-wwm-ext'
        # tokenizer_name='chinese-roberta-wwm-ext'
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.drop_last = drop_last

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = device

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )

    def collate_fn(self, data):
        '''
        data: [
        (('所以到最后把这个蛋糕做大了', '蛋糕是为了开派对准备的'), 1), 
        (('他是直接向你宣誓他的权威的,向你宣誓他的权力的', '他非常爱炫耀'), 1), 
        (('鸽群在笼中叽叽晓波的,好像也在说着私语', '鸽子很想出去'), 1), 
        ...
        (('待到静下心来,稍留些神,不用问,消息自己就来了', '要先静下心来'), 0), 
        (('妈,银行的事,我给问过了哦.', '我对银行的事一无所知。'), 2)
        ]
        '''
        sents = [i[0] for i in data]
        '''
        sents: [
        ('所以到最后把这个蛋糕做大了', '蛋糕是为了开派对准备的'), 
        ('他是直接向你宣誓他的权威的,向你宣誓他的权力的', '他非常爱炫耀'), 
        ('鸽群 在笼中叽叽晓波的,好像也在说着私语', '鸽子很想出去'), 
        ...
        ('待到静下心来,稍留些神,不用问,消息自己就来了', '要先静下心来'), 
        ('妈,银行的事,我给问过了哦.', '我对银行的事一无所知。')
        ]
        '''
        labels = [i[1] for i in data]
        '''
        labels: [1, 1, 1, ... , 0, 2]
        '''

        # 修改这里，处理两个句子的情况
        data = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(sent[0], sent[1]) for sent in sents],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_length=True
        )
        '''
        data: 
        {
        'input_ids': 
            tensor([[ 101, 2792,  809,  ...,    0,    0,    0],
            [ 101,  800, 3221,  ...,    0,    0,    0],
            [ 101, 7894, 5408,  ...,    0,    0,    0],
            ...,
            [ 101,  734,  131,  ...,    0,    0,    0],
            [ 101, 2521, 1168,  ...,    0,    0,    0],
            [ 101, 1968,  117,  ...,    0,    0,    0]]), 
        
        'token_type_ids': 
            tensor([[0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]), 
        'attention_mask': 
            tensor([[1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]]), 
        'length': 
            tensor([128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
            128, 128, 128, 128])
        }

        '''
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        return input_ids, attention_mask, token_type_ids, labels

    # 类的实例可以像迭代器一样被迭代
    # 迭代底层的`DataLoader`实例中的数据，并且通过`yield`语句将数据传递出去
    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)

