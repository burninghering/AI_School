import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from custom_dataset import CustomDataset
from custom_dataset import CLASS_NAME

LABEL_NAME = {0: 'accept', 1:'defect'}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.__dict__['resnet18'](pretrained=False, num_classes=2)

    model_path = r'' # 모델 파일 경로
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    batch_size = 4

    test_root = r'' # 테스트 데이터셋 폴더 경로
    test_dataset = CustomDataset(test_root, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    preds = {}
    labels = CLASS_NAME.keys()
    for i in labels:
        preds[i] = {}
        for j in labels:
            preds[i][j] = 0

    model.eval()
    with torch.no_grad():
        for items in test_dataloader:
            labels = items['label']
            images = items['image'].to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)


            for idx, predict in enumerate(predictions):
                predict = predict.item()

                label = labels[idx].item()
                preds[LABEL_NAME[label]][LABEL_NAME[predict]] += 1


    for key in preds.keys():
        print(key, preds[key])

    tp = preds['defect']['defect']
    fp = preds['accept']['defect']
    tn = preds['accept']['accept']
    fn = preds['defect']['accept']
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}')

if __name__ == '__main__':
    main()
