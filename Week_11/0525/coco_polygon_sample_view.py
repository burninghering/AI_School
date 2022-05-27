import os
import json
import cv2

json_path = "./json_data/instances_polygon.json"

with open(json_path, "r") as f:
    coco_info = json.load(f)

assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 정보 수집
categories = dict()
for category in coco_info['categories']:
    categories[category["id"]] = category["name"]

# print("categories info >> ", categories)

# annotaiton 정보
ann_info = dict()
for annotation in coco_info['annotations']:
    # print("annotation >> ", annotation)
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    if image_id not in ann_info:
        ann_info[image_id] = {
            "boxes": [bbox], "segmentation": [segmentation],
            "categories": [category_id]
        }
    else:
        ann_info[image_id]["boxes"].append(bbox)
        ann_info[image_id]["segmentation"].append(segmentation)
        ann_info[image_id]["categories"].append(categories[category_id])

print(ann_info)
