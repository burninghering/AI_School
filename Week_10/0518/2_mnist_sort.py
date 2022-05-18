import os
import shutil
from tqdm import tqdm

def main():
    image_root = r'C:\Users\user\ai_school\220516_20\18_minst\mnist'
    save_root = r'C:\Users\user\ai_school\220516_20\18_minst\label'

    image_paths = {}
    for (path, dir, files) in os.walk(image_root):
        for file in files:
            image_path = os.path.join(path, file)
            label = file.split('.')[0][-1]
            if label not in image_paths.keys():
                image_paths[label] = []
            image_paths[label].append(image_path) #라벨에다가 이미지 경로 넣어줌

    for label in image_paths.keys():
        path = os.path.join(save_root, label) #라벨에 대한 디렉터리 만들어주기
        os.makedirs(path, exist_ok=True) 

    for label in image_paths.keys(): #저장해놨던 경로들 읽어와서
        for path in tqdm(image_paths[label], desc=f'{label}: '):
            filename = os.path.basename(path)
            save_path = os.path.join(save_root, label, filename)
            shutil.copy(path, save_path)

if __name__ == '__main__':
    main()