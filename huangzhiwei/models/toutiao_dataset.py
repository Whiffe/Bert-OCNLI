from torch.utils.data import Dataset


class ToutiaoDataset(Dataset):
    def __init__(
        self,
        data_path,
        ignore_classes=[5, 11],
    ):
        self.data_path = data_path
        self.ignore_classes = ignore_classes
        self._get_data()
        self._get_labels()
        self._get_keywords()


    def _get_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        
        for i in range(len(data)):
            data[i] = data[i].strip().split('_!_')
            data[i] = {
                'news_id': data[i][0],
                'class_id': int(data[i][1][1:]),
                'class_name': data[i][2],
                'title': data[i][3],
                'keywords': data[i][4].split(',')
            }

        if self.ignore_classes and len(self.ignore_classes) > 0:
            max_class_id = max([item['class_id'] for item in data])
            mapping = [-1] * (max_class_id + 1)
            count = 0
            for i in range(max_class_id + 1):
                if i not in self.ignore_classes:
                    mapping[i] = count
                    count += 1

            for i in range(len(data)):
                data[i]['class_id'] = mapping[data[i]['class_id']]
            
        self.data = data
    
    def _get_labels(self):
        labels = dict()
        for item in self.data:
            labels[item['class_id']] = item['class_name']

        self.labels = [labels[i] for i in range(len(labels))]

    def _get_keywords(self):
        keywords = set()
        for item in self.data:
            keywords.update(item['keywords'])
        self.keywords = list(keywords)

    def get_labels(self):
        return self.labels
    
    def get_keywords(self):
        return self.keywords

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['title'], self.data[idx]['class_id']