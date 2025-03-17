import cv2
import pyttsx3
import time
from ultralytics import YOLO
from collections import defaultdict

engine = pyttsx3.init()
voices = engine.getProperty("voices")
for voice in voices:
    if "English" in voice.name:
        engine.setProperty("voice", voice.id)
        break

model = YOLO("yolov8x.pt")

selected_categories = [
    "person", "chair", "car", "bicycle", "motorcycle", "bus", "truck",
    "fire hydrant", "stop sign", "bench", "cat", "dog", "suitcase", "bottle",
    "wine glass", "couch", "bed", "dining table", "vase", "scissors"
]

yolo_class_mapping = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7,
    "fire hydrant": 10, "stop sign": 11, "bench": 13, "cat": 15, "dog": 16,
    "suitcase": 28, "bottle": 39, "wine glass": 40, "chair": 56, "couch": 57,
    "bed": 59, "dining table": 60, "vase": 75, "scissors": 76
}

selected_classes = [yolo_class_mapping[cat] for cat in selected_categories]

image_paths = ["./my_images/frame9.jpg", "./my_images/frame10.jpg"]

previous_boxes = {}
first_detection_reported = set()

for idx, image_path in enumerate(image_paths):
    print(f"\n rocessing {image_path}...")

    img = cv2.imread(image_path)
    orig_height, orig_width = img.shape[:2]
    resized_img = cv2.resize(img, (640, 480))

    results = model.predict(img, classes=selected_classes, conf=0.3, iou=0.7, imgsz=640)

    current_boxes = defaultdict(lambda: {"left": 0, "center": 0, "right": 0})
    movement_speech = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 目标框坐标
            center_x = (x1 + x2) / 2

            # 目标方位
            if center_x < orig_width / 3:
                position = "left"
            elif center_x < 2 * orig_width / 3:
                position = "center"
            else:
                position = "right"

            current_boxes[model.names[cls]][position] += 1

            # 计算缩放比例
            scale_x = 640 / orig_width
            scale_y = 480 / orig_height
            x1_resized = int(x1 * scale_x)
            x2_resized = int(x2 * scale_x)
            y1_resized = int(y1 * scale_y)
            y2_resized = int(y2 * scale_y)

            # 画框
            label = f"{model.names[cls]}"
            color = (0, 255, 0) if model.names[cls] == "person" else (255, 0, 0)
            cv2.rectangle(resized_img, (x1_resized, y1_resized), (x2_resized, y2_resized), color, 2)
            cv2.putText(resized_img, label, (x1_resized, y1_resized - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 计算目标变化趋势 (仅在第二张图片及之后)
            obj_key = f"{model.names[cls]}-{position}"
            area = (x2 - x1) * (y2 - y1)

            if idx > 0 and obj_key in previous_boxes:  # 仅在第二张图片及之后计算
                prev_area = previous_boxes[obj_key]
                size_change = (area - prev_area) / prev_area

                if size_change > 0.1:
                    movement_speech.append(f"You are approaching a {model.names[cls]} on your {position}.")
                elif size_change < -0.1:
                    movement_speech.append(f"You are moving away from a {model.names[cls]} on your {position}.")

            previous_boxes[obj_key] = area

    # 生成语音提示
    first_detection_speech = []
    for obj_name, positions in current_boxes.items():
        total_count = sum(positions.values())
        if total_count == 0:
            continue

        position_counts = {k: v for k, v in positions.items() if v > 0}
        speech_key = f"{obj_name}-{str(position_counts)}"
        if speech_key not in first_detection_reported:
            first_detection_reported.add(speech_key)
            if len(position_counts) == 1:
                pos = list(position_counts.keys())[0]
                first_detection_speech.append(f"{total_count} {obj_name}s detected on your {pos}.")
            else:
                position_desc = ", ".join(f"{v} on your {k}" for k, v in position_counts.items())
                first_detection_speech.append(f"{total_count} {obj_name}s detected: {position_desc}.")

    # 组合最终语音
    final_speech = " ".join(first_detection_speech + movement_speech)

    # 显示结果
    cv2.imshow("Object Detection", resized_img)
    cv2.moveWindow("Object Detection", 100, 100)

    print(f"Speaking: {final_speech}")
    engine.say(final_speech)
    engine.runAndWait()

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    time.sleep(1)

cv2.destroyAllWindows()
