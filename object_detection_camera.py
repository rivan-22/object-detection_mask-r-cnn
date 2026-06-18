import random
from pathlib import Path

import cv2
import numpy as np

CONF_THRESHOLD = 0.5
MASK_THRESHOLD = 0.3


def draw_box(image, class_id, confidence, left, top, right, bottom, class_mask):
    cv2.rectangle(image, (left, top), (right, bottom), (255, 178, 50), 3)

    label = f"{confidence:.2f}"
    if 1 <= class_id <= len(CLASSES):
        label = f"{CLASSES[class_id - 1]}:{label}"
    elif 0 <= class_id < len(CLASSES):
        label = f"{CLASSES[class_id]}:{label}"
    else:
        label = f"class_{class_id}:{label}"

    label_size, base_line = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    top = max(top, label_size[1])
    cv2.rectangle(
        image,
        (left, top - round(1.5 * label_size[1])),
        (left + round(1.5 * label_size[0]), top + base_line),
        (255, 255, 255),
        cv2.FILLED,
    )
    cv2.putText(
        image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1
    )

    class_mask = cv2.resize(class_mask, (right - left + 1, bottom - top + 1))
    mask = class_mask > MASK_THRESHOLD
    roi = image[top : bottom + 1, left : right + 1][mask]

    color = COLORS[random.randint(0, len(COLORS) - 1)]
    image[top : bottom + 1, left : right + 1][mask] = (
        [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi
    ).astype(np.uint8)

    contour_mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        image[top : bottom + 1, left : right + 1],
        contours,
        -1,
        color,
        3,
        cv2.LINE_8,
        hierarchy,
        100,
    )


def post_process(image, boxes, masks):
    _, num_detections = masks.shape[1], boxes.shape[2]
    image_height, image_width = image.shape[:2]

    for i in range(num_detections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]

        if score <= CONF_THRESHOLD:
            continue

        class_id = int(box[1])
        left = int(image_width * box[3])
        top = int(image_height * box[4])
        right = int(image_width * box[5])
        bottom = int(image_height * box[6])

        left = max(0, min(left, image_width - 1))
        top = max(0, min(top, image_height - 1))
        right = max(0, min(right, image_width - 1))
        bottom = max(0, min(bottom, image_height - 1))

        if class_id >= mask.shape[0] or class_id < 0:
            continue

        class_mask = mask[class_id]
        draw_box(image, class_id, score, left, top, right, bottom, class_mask)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CLASSES_FILE = MODELS_DIR / "mscoco_labels.names"
with open(CLASSES_FILE, "rt") as file:
    CLASSES = file.read().rstrip("\n").split("\n")

TEXT_GRAPH = MODELS_DIR / "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
MODEL_WEIGHTS = MODELS_DIR / "frozen_inference_graph.pb"
NET = cv2.dnn.readNetFromTensorflow(str(MODEL_WEIGHTS), str(TEXT_GRAPH))
NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

COLORS_FILE = MODELS_DIR / "colors.txt"
with open(COLORS_FILE, "rt") as file:
    colors_str = file.read().rstrip("\n").split("\n")
COLORS = [np.array([float(c) for c in color.split()]) for color in colors_str]


def open_camera():
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    cap.release()

    raise RuntimeError(
        "Camera tidak bisa dibuka. Di macOS, beri izin Camera untuk Terminal/iTerm/"
        "VS Code di System Settings > Privacy & Security > Camera, lalu jalankan ulang."
    )


cap = open_camera()

print("Camera aktif. Tekan 'q' untuk keluar.")

while True:
    has_frame, frame = cap.read()
    if not has_frame:
        print("Frame dari camera gagal dibaca.")
        break

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    NET.setInput(blob)
    boxes, masks = NET.forward(["detection_out_final", "detection_masks"])
    post_process(frame, boxes, masks)

    cv2.imshow("Object Detection Camera - Mask R-CNN", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
