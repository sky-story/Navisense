import cv2
import numpy as np
import time
import pandas as pd
import itertools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载 COCO 数据集
print("\n🚀 Loading COCO dataset...")
coco = COCO("D:/dev/Read_dataset/annotations/instances_val2017.json")

selected_categories = ["person", "chair", "car", "bicycle", "motorcycle", "bus", "truck", 
                       "fire hydrant", "stop sign", "bench", "cat", "dog", "suitcase", "bottle",
                       "wine glass", "couch", "bed", "dining table", "vase", "scissors"]

# 1. MobileNet-SSD 21 类别 (ID 0-20，其中 0 = background)
mobilenet_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# 2. 只映射 `selected_categories` 中的类别
mobilenet_to_coco = {}
for mobilenet_id, mobilenet_name in enumerate(mobilenet_classes):
    if mobilenet_name == "background" or mobilenet_name not in selected_categories:
        continue  # 只映射 `selected_categories` 里的类别
    coco_ids = coco.getCatIds(catNms=[mobilenet_name])  # 通过 COCO API 查找类别 ID
    if coco_ids:
        mobilenet_to_coco[mobilenet_id] = coco_ids[0]  # 取第一个匹配的类别 ID

# 3. 打印修正后的类别映射
print("\n📌 修正后的 MobileNet-SSD → COCO 类别映射 (仅限 selected_categories):")
for mobilenet_id, coco_id in mobilenet_to_coco.items():
    mobilenet_name = mobilenet_classes[mobilenet_id]  # 获取 MobileNet-SSD 类别名称
    coco_name = coco.loadCats([coco_id])[0]['name']  # 获取 COCO 类别名称
    print(f"🔄 MobileNet-SSD ID {mobilenet_id} ('{mobilenet_name}') → COCO ID {coco_id} ('{coco_name}')")

print("\n✅ MobileNet-SSD to COCO category mapping corrected!")

# 4. 选择 COCO 2017 val 数据集中包含这些类别的图片
selected_cat_ids = coco.getCatIds(catNms=[mobilenet_classes[mobilenet_id] for mobilenet_id in mobilenet_to_coco.keys()])
selected_img_ids = list(set(itertools.chain.from_iterable(
    [coco.getImgIds(catIds=[cat]) for cat in selected_cat_ids]
)))

print(f"\n✅ Found {len(selected_img_ids)} images containing selected MobileNet-SSD categories.")

# 5. 加载 MobileNet-SSD 预训练模型
print("\n🔍 Loading MobileNet-SSD model...")
mobilenet_net = cv2.dnn.readNetFromCaffe(
    "E:/GIX course/513 Signal processing/final project/deploy.prototxt", 
    "E:/GIX course/513 Signal processing/final project/mobilenet_iter_73000.caffemodel"
)
print("✅ MobileNet-SSD Model Loaded Successfully!")

# 6. 目标检测函数 (MobileNet-SSD)
def detect_objects_mobilenet(frame, conf_threshold=0.4):
    """ 使用 MobileNet-SSD 进行目标检测，并转换类别 ID """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    mobilenet_net.setInput(blob)
    detections = mobilenet_net.forward()

    detected_results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        mobilenet_class_id = int(detections[0, 0, i, 1])

        if confidence > conf_threshold and mobilenet_class_id in mobilenet_to_coco:
            coco_class_id = mobilenet_to_coco[mobilenet_class_id]  # 🔹 映射到 COCO 类别
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")

            print(f"\n🔍 Detected: {mobilenet_classes[mobilenet_class_id]} (MobileNet-SSD ID {mobilenet_class_id}) "
                  f"→ {coco_class_id} (COCO ID) | Confidence: {confidence:.2f}")
            print(f"   Bounding Box: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})")

            detected_results.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": confidence,
                "category_id": coco_class_id
            })

    return detected_results

# 7. 评估 MobileNet-SSD
results_data = []
num_samples = len(selected_img_ids)
conf_values = [0.3]  
iou_values = [0.5]  

print("\n🚀 Running MobileNet-SSD Evaluation on COCO2017 val set...")

for conf, iou in itertools.product(conf_values, iou_values):
    print(f"\n🔍 Evaluating MobileNet-SSD with conf={conf}, iou={iou}...")
    
    total_image_time = 0
    total_object_time = 0
    total_objects = 0
    detected_results = []

    for img_id in selected_img_ids[:num_samples]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"D:/dev/Read_dataset/val2017/{img_info['file_name']}"
        frame = cv2.imread(img_path)

        start_time = time.time()
        detections = detect_objects_mobilenet(frame, conf_threshold=conf)
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

        avg_image_time = total_image_time / len(selected_img_ids) if len(selected_img_ids) > 0 else 0
        avg_object_time = total_object_time / total_objects if total_objects > 0 else 0

        results_data.append({
            "Model": "mobilenet-ssd",
            "Conf": conf,
            "IoU": iou,
            "mAP50": mAP50,
            "mAP50-95": mAP50_95,
            "Avg Image Time (ms)": avg_image_time,
            "Avg Object Time (ms)": avg_object_time
        })


df = pd.DataFrame(results_data)
df.to_csv("mobilenet_coco_evaluation_results.csv", index=False)
print("\n✅ Evaluation results saved to 'mobilenet_coco_evaluation_results.csv'.")
