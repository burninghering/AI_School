import os
import json
import cv2

json_path="./json_data/instances_default.json"

#json 열기
with open(json_path,"r") as f:
    coco_info=json.load(f)

# print(coco_info)

assert len(coco_info) > 0, "파일 읽기 실패"

#카테고리 정보 수정
categories=dict()
for category in coco_info['categories']:
    categories[category["id"]]=category["name"]

# print("categories info >> ",categories)
#categories info >>  {1: 'rose'}

#원래 카테고리 정보는 
    # "categories":[
    #    {
    #       "id":1,
    #       "name":"rose",
    #       "supercategory":""
    #    }
    # ]

#어노테이션 정보들을 넣기 
ann_info = dict()
for annotation in coco_info['annotations']: #coco_info에 있던 원래 어노테이션 정보들(csv 형태)을 파이썬 dict 형태로 만들어주는 것 같다.
    print("annotation >> ", annotation)
    #annotation >>  {'id': 1, 'image_id': 1, 'category_id': 1, 'segmentation': [], 'area': 151291.13280000002, 'bbox': [299.46, 112.03, 395.76, 382.28], 'iscrowd': 0, 'attributes': {'occluded': False}}
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]

    if image_id not in ann_info: #image_id가 dict에 없으면
        ann_info[image_id] = { #dict 형태로 넣어주고
            "boxes": [bbox], "categories": [category_id]
        }
    else: #있으면 덮어씌우라는것같은데 
        ann_info[image_id]["boxes"].append(bbox) 
        ann_info[image_id]["categories"].append(categories[category_id])

# print(ann_info)
#{1: {'boxes': [[299.46, 112.03, 395.76, 382.28]], 'categories': [1]}}





#    "images":[
#       {
#          "id":1,
#          "width":945,
#          "height":629,
#          "file_name":"image.jpeg",
#          "license":0,
#          "flickr_url":"",
#          "coco_url":"",
#          "date_captured":0
#       }
#    ]

for image_info in coco_info['images']:
    filename = image_info["file_name"]
    height = image_info["height"]
    width = image_info["width"]
    img_id = image_info["id"]

    # print(filename,height,width) #이미지, 높이, 넓이
    #image.jpeg 629 945




    #경로 만들어주기
    file_path = os.path.join("./0525_image_data/", filename)
    # print(file_path)
    #./0525_image_data/image.jpeg

    #이미지 읽기
    img = cv2.imread(file_path)

    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue

    # print(annotation)
    # {'boxes': [[299.46, 112.03, 395.76, 382.28]], 'categories': [1]}

    for bbox, category in zip(annotation["boxes"], annotation["categories"]):
        x1, y1, w, h = bbox

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        #원본 이미지 남겨놓기
        org_img = img.copy()

        # if category==1:
        #     category="ros"


        text_img = cv2.putText(img, str(category),
                               (int(x1), int(y1)-10), font, fontScale, color, thickness, cv2.LINE_AA)

        #직사각형 그리기
        rec_img = cv2.rectangle(text_img, (int(x1), int(
            y1)), (int(x1+w), int(y1+h)), (255, 0, 255), 2)
        cv2.imshow("test", rec_img)
        cv2.waitKey(0)