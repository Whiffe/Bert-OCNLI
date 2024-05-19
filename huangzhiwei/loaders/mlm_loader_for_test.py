import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class MLMDataLoader:
    def __init__(
        self,
        dataset,
        batch_size=32,
        max_length=128,
        shuffle=False,
        drop_last=False,
        device=None,
        tokenizer_name='chinese-bert-wwm-ext'
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
        sents = data

        data = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(sent[0], sent[1]) for sent in sents],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_length=True
        )
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)

        return input_ids, attention_mask, token_type_ids

    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)
