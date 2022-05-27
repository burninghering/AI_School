from PIL import Image
from torch.utils.data import Dataset
import os


CLASS_NAME = {'accept': 0, 'defect': 1}

class CustomDataset(Dataset):
    def __init__(self, image_root, transform, mode='train'):
        self.image_paths = get_img_paths(image_root)
        self.labels = []

        if mode == 'train':
            for image_path in self.image_paths:
                label = image_path.split('\\')[-2]
                self.labels.append(CLASS_NAME[label])
        else:
            self.labels = [-1 for i in range(len(self.image_paths))]

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return {'label': label, 'image':image, 'path':image_path}


def get_img_paths(image_root):
    image_paths = []
    for (path, dir, files) in os.walk(image_root):
        for file in files:
            ext = file.split('.')[-1]
            if ext in ['jpg', 'png']:
                image_path = os.path.join(path, file)
                image_paths.append(image_path)
    return image_paths