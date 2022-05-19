import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models


from cnn_model import ConvNet
from custom_dataset import CustomDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model 지정하기
    model = ConvNet()
    model = model.to(device)

    # loss function, optimizer(learning rate) 지정하기
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # batch size, epoch 지정
    batch_size = 64
    epoch = 10

    # train용 dataset 불러오기
    train_root = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\train'
    train_dataset = CustomDataset(train_root, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # valid용 dataset 불러오기
    valid_root = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\valid'
    valid_dataset = CustomDataset(valid_root, transform=transforms.ToTensor())
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    save_root = r'C:\Users\user\ai_school\220516_20\18_minst\_dataset\model_pth'
    os.makedirs(save_root, exist_ok=True)

    best_accuracy = 0

    # 학습 및 validation
    for epoch_idx in range(epoch):

        # train
        model.train()
        losses = 0
        for item in tqdm(train_dataloader):
            labels = item['label'].to(device)
            images = item['image'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            losses += loss

        # valid
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for item in valid_dataloader:
                labels = item['label'].to(device)
                images = item['image'].to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (labels == predictions).sum()

        avg_loss = losses / batch_size
        accuracy = correct / total * 100
        print(f'epoch: {epoch_idx} - Accuracy: {accuracy}, Avg_loss: {avg_loss}')
        torch.save(model.state_dict(), f'{save_root}/{epoch_idx}.pth')
        
        # 정확도가 제일 높은 모델 'best.pth' 이름으로 저장하기
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'{save_root}/best.pth')


if __name__ == '__main__':
    main()