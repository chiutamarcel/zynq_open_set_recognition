from include.common import *
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class EuroSAT(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx]['Filename']

        if (self.data.iloc[idx]['Label'] in known_labels):
            known = 1
        else:
            known = 0

        label = self.data.iloc[idx]['Label']

        img_path = os.path.join(self.root_dir, img_rel_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, known