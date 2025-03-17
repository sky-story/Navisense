from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import time
import pandas as pd
import itertools
import numpy as np

if __name__ == '__main__':
    print("\n🚀 Loading COCO dataset...")
    coco = COCO("D:/dev/Read_dataset/annotations/instances_val2017.json")

    selected_categories = ["person", "chair", "car", "bicycle", "motorcycle", "bus", "truck", 
                           "fire hydrant", "stop sign", "bench", "cat", "dog", "suitcase", "bottle",
                           "wine glass", "couch", "bed", "dining table", "vase", "scissors"]

    # YOLO 类别索引
    yolo_class_mapping = {
        "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, 
        "fire hydrant": 10, "stop sign": 11, "bench": 13, "cat": 15, "dog": 16, 
        "suitcase": 28, "bottle": 39, "wine glass": 40, "chair": 56, "couch": 57, 
        "bed": 59, "dining table": 60, "vase": 75, "scissors": 76
    }

    selected_classes = [yolo_class_mapping[cat] for cat in selected_categories]
    print(f"\n✅ YOLO will detect these classes: {selected_classes}")

    selected_cat_ids = coco.getCatIds(catNms=selected_categories)
    print(f"✅ COCO Category IDs selected: {selected_cat_ids}")

    selected_img_ids = list(set(itertools.chain.from_iterable([coco.getImgIds(catIds=[cat]) for cat in selected_cat_ids])))
    print(f"✅ Found {len(selected_img_ids)} images containing selected categories.")

    print("\n📌 Checking Category Mappings...")

    # YOLO 类别索引到 COCO ID 的映射
    yolo_class_to_coco_id = {}

    for cat in selected_categories:
        coco_id = coco.getCatIds(catNms=[cat])[0] 
        yolo_id = yolo_class_mapping.get(cat, None)

        coco_cat_name = coco.loadCats([coco_id])[0]['name'] if coco_id else "Unknown"
        yolo_cat_name = next((k for k, v in yolo_class_mapping.items() if v == yolo_id), "Unknown")

        print(f"YOLO Index {yolo_id} ('{yolo_cat_name}') → COCO Index {coco_id} ('{coco_cat_name}')")

        # 存储到映射字典
        yolo_class_to_coco_id[yolo_id] = coco_id

    print(f"\n✅ Final YOLO to COCO Category Mapping: {yolo_class_to_coco_id}")

    results_data = []

    # 选择要测试的 YOLO 模型
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    conf_values = [0.3, 0.5, 0.7]
    iou_values = [0.3, 0.5, 0.7]

    num_samples = len(selected_img_ids)

    for model_name, conf, iou in itertools.product(models, conf_values, iou_values):
        print(f"\n🔍 Evaluating {model_name} with conf={conf}, iou={iou}...")
        model = YOLO(model_name)
        detected_results = []

        total_image_time = 0
        total_object_time = 0
        total_objects = 0
        processed_images = 0

        for img_id in selected_img_ids[:num_samples]:
            img_info = coco.loadImgs(img_id)[0]
            img_path = f"D:/dev/Read_dataset/val2017/{img_info['file_name']}"

            start_time = time.time()
            results = model.predict(img_path, conf=conf, iou=iou, imgsz=640, classes=selected_classes)
            image_time = (time.time() - start_time) * 1000  # ms
            total_image_time += image_time

            num_objects_in_image = len(results[0].boxes) if results[0].boxes else 0
            total_objects += num_objects_in_image

            if num_objects_in_image > 0:
                avg_object_time_per_image = image_time / num_objects_in_image
                total_object_time += avg_object_time_per_image * num_objects_in_image

            processed_images += 1
            remaining_images = num_samples - processed_images

            print(f"\n🖼️ Processed {processed_images}/{num_samples} images, Remaining: {remaining_images}")
            print(f"   ⏳ Image {img_id} processed in {image_time:.2f} ms")
            print(f"   📊 Detected {num_objects_in_image} objects")

            for box in results[0].boxes:
                class_index = int(box.cls)
                if class_index not in yolo_class_to_coco_id:
                    continue

                coco_category_id = yolo_class_to_coco_id[class_index]
                x_center, y_center, w, h = box.xywh[0]
                x_min = x_center - w / 2
                y_min = y_center - h / 2

                detected_results.append({
                    "image_id": img_id,
                    "category_id": coco_category_id,
                    "bbox": [float(x_min), float(y_min), float(w), float(h)],
                    "score": float(box.conf)
                })

        avg_image_time = total_image_time / num_samples if num_samples > 0 else 0
        avg_object_time = total_object_time / total_objects if total_objects > 0 else 0

        print(f"\n📊 Total detections collected: {len(detected_results)}")
        print(f"⏳ Average Image Processing Time: {avg_image_time:.2f} ms")
        print(f"⏳ Average Object Processing Time: {avg_object_time:.2f} ms")

        if detected_results:
            coco_dt = coco.loadRes(detected_results)
            coco_eval = COCOeval(coco, coco_dt, "bbox")
            coco_eval.params.imgIds = selected_img_ids
            coco_eval.params.catIds = list(set([d["category_id"] for d in detected_results]))

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precision = coco_eval.stats[1]  
            recall = coco_eval.stats[8]
            mAP50 = coco_eval.stats[1]  
            mAP50_95 = coco_eval.stats[0]  

            results_data.append({
                "Model": model_name,
                "Conf": conf,
                "IoU": iou,
                "mAP50": mAP50,
                "mAP50-95": mAP50_95,
                "Avg Image Time (ms)": avg_image_time,
                "Avg Object Time (ms)": avg_object_time
            })

    best_result = max(results_data, key=lambda x: x["mAP50-95"])
    print(f"\n🏆 Best Model Configuration: {best_result}")

    df = pd.DataFrame(results_data)
    df.to_csv("yolo_coco_evaluation_results.csv", index=False)
    print("\n✅ Evaluation results saved to 'yolo_coco_evaluation_results.csv'.")
