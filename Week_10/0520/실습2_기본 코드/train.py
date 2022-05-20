import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from custom_dataset import CustomDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model 지정하기 # vgg11, vgg13, resnet18, resnet34 등 pytorch에서 지원하는 모델 사용
    # 관련 깃허브: https://github.com/pytorch/vision/tree/main/torchvision/models
    model = models.__dict__['resnet18'](pretrained=False, num_classes=2)
    model = model.to(device)

    

if __name__ == '__main__':
    main()