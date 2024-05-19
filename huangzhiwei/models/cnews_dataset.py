from torch.utils.data import Dataset


class CnewsDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_map={
            "体育": 0,
            "娱乐": 1,
            "家居": 2,
            "房产": 3,
            "教育": 4,
            "时尚": 5,
            "时政": 6,
            "游戏": 7,
            "科技": 8,
            "财经": 9,
        }
    ):
        self.data_path = data_path
        self.label_map = label_map
        self._get_data()

    def _get_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        
        self.data = []
        for line in data:
            line = line.strip().split('\t')
            news = line[1]
            label = line[0]
            if label in self.label_map:
                class_id = self.label_map[label]
                self.data.append((news, class_id))

    def __len__(self):
        return len(self.data)


    def get_labels(self):
        labels = dict()
        for key, value in self.label_map.items():
            labels[value] = key
        return labels


    
    def __getitem__(self, idx):
        return self.data[idx]
