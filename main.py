import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import time

# Init cv2
cap = cv2.VideoCapture('Vid2.mp4')
net = cv2.dnn.readNetFromONNX("model.onnx")

# Only 2 Classes For Now
classes = ["bike", "head"]

# Resnet Model
helmet_classification_threshold = 0.3
resnet_model = models.resnet18(pretrained=True)
resnet_model.eval()

# Resnet Transformations
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

while True:
    img = cap.read()[1]
    if img is None:
        break

    img = cv2.resize(img, (640, 640))
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width / 640
    y_scale = img_height / 640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.3:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.3 and classes[ind] == "bike":
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx - w / 2) * x_scale)
                y1 = int((cy - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    for i in indices:
        time_stamp = str(time.time())
        x1, y1, w, h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        frame = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 1)  # Adjust the thickness here

        # Ensure that indices are integers
        y1, y2, x1, x2 = int(y1), int(y1 + h), int(x1), int(x1 + w)

        roi = img[y1:y2, x1:x2]

        if roi.shape[0] > 0 and roi.shape[1] > 0:
            resnet_input = resnet_transform(Image.fromarray(roi))
            resnet_input = resnet_input.unsqueeze(0)
            with torch.no_grad():
                resnet_output = resnet_model(resnet_input)

            # Helmet and Non-Helmet Classification
            helmet_confidence = torch.softmax(resnet_output, dim=1)[0][0].item()
            if helmet_confidence > helmet_classification_threshold:
                print("Helmet detected with confidence:", helmet_confidence)
                # Perform further processing for helmet detection
            else:
                print("Non-helmet detected with confidence:", 1 - helmet_confidence)
                cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', roi)
                # Perform further processing for non-helmet detection
        else:
            print("Invalid ROI, not saving image.")

        frame = cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("VIDEO", img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

