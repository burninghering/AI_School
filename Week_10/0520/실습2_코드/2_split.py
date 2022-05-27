import random
import os
import shutil
from tqdm import tqdm

def main():
    image_root = r'' # accept/defect로 분류된 데이터 폴더 경로
    save_root = r''

    image_paths = {}
    # { label1: [path1, path2, ...],
    #   label2: [path21, ...],
    #   ...
    # }

    for (path, dir, files) in os.walk(image_root):
        for file in files:
            image_path = os.path.join(path, file)
            label = os.path.basename(path)
            if label not in image_paths.keys():
                image_paths[label] = []

            image_paths[label].append(image_path)

    use_type = ['train', 'test']
    for label in image_paths.keys():
        for use in use_type:
            path = os.path.join(save_root, use, label)
            os.makedirs(path, exist_ok=True)

    for label in image_paths.keys():
        paths = image_paths[label]
        random.shuffle(paths)

        idx_train = int(len(paths) * 0.9)
        for path in tqdm(paths[:idx_train], desc=f'{label}-train'):
            filename = os.path.basename(path)
            save_path = os.path.join(save_root, 'train', label, filename)
            shutil.copy2(path, save_path)

        for path in tqdm(paths[idx_train:], desc=f'{label}-test'):
            filename = os.path.basename(path)
            save_path = os.path.join(save_root, 'test', label, filename)
            shutil.copy2(path, save_path)


if __name__ == '__main__':
    main()