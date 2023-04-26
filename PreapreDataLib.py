import os
import shutil
import re
import cv2
import numpy as np

annotation_id = 0

def find_contours(sub_mask):
    _, thresh = cv2.threshold(sub_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


def create_category_annotation(category_dict):
    category_list = []
    for key, value in category_dict.items():
        category = {"id": int(value), "name": key, "supercategory": key}
        category_list.append(category)
    return category_list


class CocoImage:
    def __init__(self, image_id: int, w: int, h: int, name: str):
        self.id = image_id
        self.width = w
        self.height = h
        self.file_name = name

    def get_as_dict(self):
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.file_name,
        }


class CocoAnnotation:
    def __init__(self, ann_id: int, image_id: int, category_id: int, contour):
        self.iscrowd = 0
        self.id = ann_id
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = cv2.boundingRect(contour)
        self.area = cv2.contourArea(contour)
        self.segmentation = [contour.flatten().tolist()]

    def get_as_dict(self):
        return {
            "iscrowd": self.iscrowd,
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "area": self.area,
            "segmentation": self.segmentation,
        }


class CocoDataset:
    def __init__(self, categories: dict[str, int]):
        self.info = dict()
        self.licenses = []
        self.images = [dict()]
        self.categories = create_category_annotation(categories)
        self.annotations = [dict()]

    def get_as_dict(self):
        return {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "categories": self.categories,
            "annotations": self.annotations,
        }


def process_image_and_masks(src_dir, dest_dir, ann_file: CocoDataset):
    for img in os.listdir(src_dir):
        image_id = int(img.split("_")[0])
        IMG_PATH = os.path.join(src_dir, img)
        image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
        if re.match(".*RAW.*", img):
            height, width = image.shape
            image_coco = CocoImage(image_id, width, height, img)
            ann_file.images.append(image_coco.get_as_dict())
            shutil.copy(IMG_PATH, dest_dir)
        else:
            if cv2.countNonZero(image) != 0:
                category_id = 0
                if not re.match(".*LANE.*", img):
                    if re.match(".*CRACK.*", img):
                        category_id = next(item for item in ann_file.categories if item["name"] == "crack")["id"]
                    if re.match(".*POTHOLE.*", img):
                        category_id = next(item for item in ann_file.categories if item["name"] == "pothole")["id"]
                    # if re.match(".*LANE.*", img):
                    #     category_id = next(item for item in ann_file.categories if item["name"] == "lane")["id"]
                    assert category_id > 0
                    contours = find_contours(image)
                    for contour in contours:
                        global annotation_id
                        annotation = CocoAnnotation(annotation_id, image_id, category_id, contour)
                        if annotation.area > 0:
                            ann_file.annotations.append(annotation.get_as_dict())
                            annotation_id += 1


def create_folder_if_not_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return True # папка была создана
    return False # папка не была создана
