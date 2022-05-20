import os
import shutil
from tqdm import tqdm

def main():
    image_root = r'' # 이미지가 섞여 있는 폴더 경로
    save_root = r'' # accept, defect 분류할 폴더 경로

    image_paths = {}

    for (path, dir, files) in os.walk(image_root):
        for file in files:
            image_path = os.path.join(path, file)
            label = file.split('.')[0].split('_')[-1]

            if label == 'ok': label = 'accept'
            if label == 'def': label = 'defect'
            if label not in image_paths.keys():
                image_paths[label] = []

            image_paths[label].append(image_path)


    for label in image_paths.keys():
        path = os.path.join(save_root, label)
        os.makedirs(path, exist_ok=True)


    for label in image_paths.keys():
        for path in tqdm(image_paths[label], desc=f'{label}'):
            save_path = os.path.join(save_root, label, os.path.basename(path))
            shutil.copy2(path, save_path)


if __name__ == '__main__':
    main()