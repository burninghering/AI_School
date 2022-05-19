from PIL import Image
from torch.utils.data import Dataset
import os

CLASS_NAME = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
              '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
class CustomDataset(Dataset):
    def __init__(self, image_root, transform, mode='train'):
        # 전처리
        # 사용할 이미지들의 경로 / 라벨 관련된 내용을 모두 지정
        self.image_paths = get_img_paths(image_root)
        self.labels = []

        if mode == 'train':
            for image_path in self.image_paths:
                # C:\Users\user\ai_school\220516_20\18_minst\_dataset\train\0\image.png
                label = image_path.split('\\')[-2]
                self.labels.append(CLASS_NAME[label])
        else:
            self.labels = [-1 for i in range(len(self.image_paths))]

        self.transform = transform

    def __len__(self): # 데이터셋 크기 반환
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 읽어서 라벨과 반환 / 학습에 들어갈 정보 반환
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        image = self.transform(image)

        return {'label': label, 'image': image, 'path': image_path}


def get_img_paths(image_root):
    image_paths = []
    for (path, dir, files) in os.walk(image_root):
        for file in files:
            ext = file.split('.')[-1]
            if ext in ['jpg', 'png']:
                image_path = os.path.join(path, file)
                image_paths.append(image_path)
    return image_paths
