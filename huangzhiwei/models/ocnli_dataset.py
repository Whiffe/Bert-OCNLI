from torch.utils.data import Dataset


class OcnliDataset(Dataset):
    def __init__(
            self,
            data_path,
            label_map={
                "0": 0,
                "1": 1,
                "2": 2,
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
            sent1 = line[1]
            sent2 = line[2]
            label = line[3]
            if label in self.label_map:
                class_id = self.label_map[label]
                self.data.append(((sent1, sent2), class_id))

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labels = dict()
        for key, value in self.label_map.items():
            labels[value] = key
        return labels

    def __getitem__(self, idx):
        return self.data[idx]
