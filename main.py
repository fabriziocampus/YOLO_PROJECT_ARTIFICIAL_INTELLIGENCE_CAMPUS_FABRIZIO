import cv2
import numpy as np


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45


FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def draw_label(input_image, label, left, top):

    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]

    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);

    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


if __name__ == '__main__':

    classesFile = "YOLOv5/labels.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    image = cv2.imread('YOLOv5/image.jpg')
    class_ids = []
    confidences = []
    boxes = []

    modelWeights = "YOLOv5/models/yolov5m.onnx"
    net = cv2.dnn.readNet(modelWeights)

    blob = cv2.dnn.blobFromImage(image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    rows = detections[0].shape[1]

    image_height, image_width = image.copy().shape[:2]

    x = image_width / INPUT_WIDTH
    y = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = detections[0][0][r]
        confidence = row[4]

        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            class_id = np.argmax(classes_scores)

            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x)
                top = int((cy - h / 2) * y)
                width = int(w * x)
                height = int(h * y)

                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(image, label, left, top)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    cv2.putText(image, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

    cv2.imshow('Output', image)
    cv2.waitKey(0)
