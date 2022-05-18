import shutil
import os
import random
from tqdm import tqdm

def main():
    image_root = r'C:\Users\user\ai_school\220516_20\18_minst\label'
    save_root = r'C:\Users\user\ai_school\220516_20\18_minst\dataset'

    image_paths = {} #한 라벨에 대해 이미지 경로 저장
    for (path, dir, files) in os.walk(image_root):
        for file in files:
            image_path = os.path.join(path, file)
            label = os.path.basename(path)
            if label not in image_paths.keys():
                image_paths[label] = []
            image_paths[label].append(image_path)

    use_type = ['train', 'valid', 'test'] #하위 폴더들 나오게
    for label in image_paths.keys():
        for use in use_type:
            path = os.path.join(save_root, use, label)
            os.makedirs(path, exist_ok=True)


    for label in image_paths.keys(): #train과 test가 라벨이 통일하도록 맞춰버림
        path_list = image_paths[label]
        random.shuffle(path_list) #경로들 다 섞어버리기

        idx_train = int(len(path_list) * 0.8)
        idx_valid = int(len(path_list) * 0.9)
        for path in tqdm(path_list[:idx_train], desc=f'{label}-train: '): #이미지 경로에서 순서대로 train에 집어넣고
            filename = os.path.basename(path)
            save_path = os.path.join(save_root, 'train', label, filename)
            shutil.copy(path, save_path)

        for path in tqdm(path_list[idx_train:idx_valid], desc=f'{label}-valid: '):
            filename = os.path.basename(path)
            save_path = os.path.join(save_root, 'valid', label, filename)
            shutil.copy(path, save_path)

        for path in tqdm(path_list[idx_valid:], desc=f'{label}-test: '):
            filename = os.path.basename(path)
            save_path = os.path.join(save_root, 'test', label, filename)
            shutil.copy(path, save_path)

if __name__ == '__main__':
    main()