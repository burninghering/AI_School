import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import os
import shutil

from cnn_model import ConvNet
from custom_dataset import CustomDataset
from custom_dataset import CLASS_NAME


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConvNet()

    model_path = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\model_pth\9.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    #모델 불러오는 과정 끝

    test_root = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\test'
    test_dataset = CustomDataset(test_root, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    estimate = {}
    for i in CLASS_NAME.keys():
        estimate[i] = {}
        for j in CLASS_NAME.keys():
            estimate[i][j] = 0

    model.eval()
    with torch.no_grad():
        for item in test_loader:
            labels = item['label']
            images = item['image'].to(device)
            image_paths = item['path']

            outputs = model(images) #결과중에 제일 높았던 인덱션 뽑아오기 위해 torch 사용 
            _, predictions = torch.max(outputs.data, 1)

            for idx, predict in enumerate(predictions):
                predict = predict.item()
                image_path = image_paths[idx]

                label = labels[idx].item()
                estimate[str(label)][str(predict)] += 1

    for i in estimate.keys():
        print(i, estimate[i])


if __name__ == '__main__':
    main()