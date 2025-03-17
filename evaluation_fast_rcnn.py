import torch
import torchvision
import time
import pandas as pd
import itertools
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np

# 1. 加载 COCO 数据集
print("\n🚀 Loading COCO dataset...")
coco = COCO("D:/dev/Read_dataset/annotations/instances_val2017.json")

selected_categories = ["person", "chair", "car", "bicycle", "motorcycle", "bus", "truck", 
                       "fire hydrant", "stop sign", "bench", "cat", "dog", "suitcase", "bottle",
                       "wine glass", "couch", "bed", "dining table", "vase", "scissors"]

# 2. 加载 Faster R-CNN 预训练模型
print("\n🔍 Loading Faster R-CNN model...")
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()
print("✅ Faster R-CNN Model Loaded Successfully!")

# 3. 计算 COCO 类别索引 (COCO → Faster R-CNN)
coco_to_faster_rcnn = {}
for cat in selected_categories:
    coco_id = coco.getCatIds(catNms=[cat])[0]  # 获取 COCO 类别 ID
    coco_to_faster_rcnn[coco_id] = coco_id  # Faster R-CNN 直接使用 COCO ID

# 4. 打印类别映射
print("\n📌 Faster R-CNN 类别索引映射 (仅限 selected_categories):")
for coco_id in coco_to_faster_rcnn:
    coco_name = coco.loadCats([coco_id])[0]['name']
    print(f"🔄 COCO ID {coco_id} ('{coco_name}') → Faster R-CNN ID {coco_to_faster_rcnn[coco_id]} ('{coco_name}')")

print("\n✅ Faster R-CNN category mapping complete!")

# 5. 选择 COCO 2017 val 数据集中包含这些类别的图片
selected_cat_ids = list(coco_to_faster_rcnn.keys())
selected_img_ids = list(set(itertools.chain.from_iterable(
    [coco.getImgIds(catIds=[cat]) for cat in selected_cat_ids]
)))

print(f"\n✅ Found {len(selected_img_ids)} images containing selected Faster R-CNN categories.")

# 6. 目标检测函数 (Faster R-CNN)
def detect_objects_faster_rcnn(frame, conf_threshold=0.4):
    """ 使用 Faster R-CNN 进行目标检测，并转换类别 ID """
    img_tensor = F.to_tensor(frame).unsqueeze(0)  # 转换为 PyTorch tensor

    with torch.no_grad():
        predictions = faster_rcnn_model(img_tensor)

    detected_results = []
    for i, box in enumerate(predictions[0]["boxes"]):
        score = predictions[0]["scores"][i].item()
        label = predictions[0]["labels"][i].item()

        if score > conf_threshold and label in coco_to_faster_rcnn:
            coco_class_id = coco_to_faster_rcnn[label]  # 🔹 直接使用 COCO 类别 ID
            x1, y1, x2, y2 = box.tolist()

            print(f"\n🔍 Detected: COCO ID {label} → '{coco.loadCats([coco_class_id])[0]['name']}' | Confidence: {score:.2f}")
            print(f"   Bounding Box: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})")

            detected_results.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": score,
                "category_id": coco_class_id
            })

    return detected_results

# 7. 评估 Faster R-CNN
results_data = []
num_samples = len(selected_img_ids)
conf_values = [0.3]  
iou_values = [0.5]  

print("\n🚀 Running Faster R-CNN Evaluation on COCO2017 val set...")

for conf, iou in itertools.product(conf_values, iou_values):
    print(f"\n🔍 Evaluating Faster R-CNN with conf={conf}, iou={iou}...")
    
    total_image_time = 0
    total_object_time = 0
    total_objects = 0
    detected_results = []

    for img_id in selected_img_ids[:num_samples]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"D:/dev/Read_dataset/val2017/{img_info['file_name']}"
        frame = cv2.imread(img_path)

        start_time = time.time()
        detections = detect_objects_faster_rcnn(frame, conf_threshold=conf)
        image_time = (time.time() - start_time) * 1000
        total_image_time += image_time

        for detection in detections:
            obj_start_time = time.time()
            obj_time = (time.time() - obj_start_time) * 1000
            total_object_time += obj_time
            total_objects += 1

            detected_results.append({
                "image_id": img_id,
                "category_id": detection["category_id"],
                "bbox": detection["bbox"],
                "score": detection["score"]
            })

    avg_image_time = total_image_time / len(selected_img_ids) if len(selected_img_ids) > 0 else 0
    avg_object_time = total_object_time / total_objects if total_objects > 0 else 0

    if detected_results:
        coco_dt = coco.loadRes(detected_results)
        coco_eval = COCOeval(coco, coco_dt, "bbox")
        coco_eval.params.imgIds = selected_img_ids
        coco_eval.params.catIds = list(set([d["category_id"] for d in detected_results]))

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        precision = coco_eval.stats[1]  # mAP@0.50
        recall = coco_eval.stats[8]  # Recall
        mAP50 = coco_eval.stats[1]  # mAP@0.50
        mAP50_95 = coco_eval.stats[0]  # mAP@[0.50:0.95]

        results_data.append({
            "Model": "faster-rcnn",
            "Conf": conf,
            "IoU": iou,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "Avg Image Time (ms)": avg_image_time,
            "Avg Object Time (ms)": avg_object_time
        })

df = pd.DataFrame(results_data)
df.to_csv("faster_rcnn_coco_evaluation_results.csv", index=False)
print("\n✅ Evaluation results saved to 'faster_rcnn_coco_evaluation_results.csv'.")
