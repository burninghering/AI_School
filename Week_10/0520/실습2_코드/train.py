import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from custom_dataset import CustomDataset

LABEL_NAME = {0: 'accept', 1:'defect'}

def main():
    # model 지정하기
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 가져오기
    model = models.__dict__['resnet18'](pretrained=False, num_classes=2)
    model = model.to(device)

    # loss function, optimizer(learning rate) 지정하기
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # batch size, epoch 지정
    batch_size = 16
    epoch = 10

    train_root = r'' # 학습 데이터셋 경로
    train_dataset = CustomDataset(train_root, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_root = r'' # 테스트 데이터셋 경로
    valid_dataset = CustomDataset(valid_root, transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    save_root = r'' # 모델 파일 저장 경로

    best_accuracy = 0
    best_sensitivity = 0
    best_specificity = 0
    preds = {}
    classes = LABEL_NAME.values()
    for i in classes:
        preds[i] = {}
        for j in classes:
            preds[i][j] = 0

    for epoch_idx in range(epoch):
        # 학습 및 학습 시 계산한 loss 기록
        model.train()
        losses = 0
        for step, item in enumerate(train_loader):
            labels = item['label'].to(device)
            images = item['image'].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            losses += loss

            _, predictions = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predictions == labels).sum()
            print(f'[epoch:{epoch_idx}/{epoch}][step:{step+1}/{len(train_loader)}] accuracy: {correct/total * 100}, loss: {loss}')

        # validation
        for i in classes:
            for j in classes:
                preds[i][j] = 0

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for item in valid_loader:
                labels = item['label'].to(device)
                images = item['image'].to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predictions == labels).sum()

                for idx, predict in enumerate(predictions):
                    predict = predict.cpu().item()

                    label = labels[idx].cpu().item()
                    preds[LABEL_NAME[label]][LABEL_NAME[predict]] += 1

        avg_loss = losses / batch_size

        tp = preds['defect']['defect']
        fp = preds['accept']['defect']
        tn = preds['accept']['accept']
        fn = preds['defect']['accept']
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f'epoch: {epoch_idx} - Accuracy: {accuracy*100}, loss:{avg_loss}')
        print(f'epoch: {epoch_idx} - Sensitivity: {sensitivity*100}, Specificity:{specificity*100}')

        torch.save(model.state_dict(), f'{save_root}/{epoch_idx}.pth')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'{save_root}/best.pth')
        if sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            torch.save(model.state_dict(), f'{save_root}/best_sensitivity.pth')
        if specificity > best_specificity:
            best_specificity = specificity
            torch.save(model.state_dict(), f'{save_root}/best_specificity.pth')


if __name__ == '__main__':
    main()