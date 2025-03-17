import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO
from collections import defaultdict

engine = pyttsx3.init()

voices = engine.getProperty('voices')
for voice in voices:
    if "English" in voice.name:
        engine.setProperty('voice', voice.id)
        break

model = YOLO("yolov8n.pt")

selected_classes = [0, 1, 2, 3, 5, 7, 11, 56, 59]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

previous_boxes = {}
first_detection_reported = set()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.predict(frame, classes=selected_classes, conf=0.4, iou=0.5, imgsz=640)

    orig_height, orig_width = frame.shape[:2]

    resized_img = cv2.resize(frame, (640, 480))

    current_boxes = defaultdict(lambda: {"left": 0, "center": 0, "right": 0})
    movement_speech = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            center_x = (x1 + x2) / 2

            if center_x < orig_width / 3:
                position = "left"
            elif center_x < 2 * orig_width / 3:
                position = "center"
            else:
                position = "right"

            current_boxes[model.names[cls]][position] += 1

            scale_x = 640 / orig_width
            scale_y = 480 / orig_height
            x1_resized = int(x1 * scale_x)
            x2_resized = int(x2 * scale_x)
            y1_resized = int(y1 * scale_y)
            y2_resized = int(y2 * scale_y)

            label = f"{model.names[cls]}"
            color = (0, 255, 0) if model.names[cls] == "person" else (255, 0, 0)
            cv2.rectangle(resized_img, (x1_resized, y1_resized), (x2_resized, y2_resized), color, 2)
            cv2.putText(resized_img, label, (x1_resized, y1_resized - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            obj_key = f"{model.names[cls]}-{position}"
            if obj_key in previous_boxes:
                prev_area = previous_boxes[obj_key]
                area = (x2 - x1) * (y2 - y1)
                size_change = (area - prev_area) / prev_area

                if size_change > 0.1:
                    movement_speech.append(f"You are approaching a {model.names[cls]} on your {position}.")
                elif size_change < -0.1:
                    movement_speech.append(f"You are moving away from a {model.names[cls]} on your {position}.")

            previous_boxes[obj_key] = (x2 - x1) * (y2 - y1)

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

    final_speech = " ".join(first_detection_speech + movement_speech)

    cv2.imshow("Object Detection", resized_img)
    cv2.moveWindow("Object Detection", 100, 100)

    print(f"Speaking: {final_speech}")
    
    if final_speech:
        speech_thread = threading.Thread(target=speak, args=(final_speech,), daemon=True)
        speech_thread.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program exited successfully.")
