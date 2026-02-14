import cv2
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ALWAYS LOAD CHART =====
base_path = os.path.dirname(os.path.abspath(__file__))
chart_path = os.path.join(base_path, "HAND SIGN REFERENCE.jpg")

chart = cv2.imread(chart_path)

if chart is None:
    print(f"⚠ chart NOT FOUND at: {chart_path}")
else:
    chart = cv2.resize(chart, (500, 500))
    cv2.imshow("ASL CHART", chart)


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgHeight, imgWidth, _ = img.shape

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            #BIG CLEAR TEXT
            text = labels[index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2.2
            thickness = 4

            (tw, th), _ = cv2.getTextSize(text, font, fontScale, thickness)

            box_x1 = x - offset
            box_y1 = y - offset - 80
            box_x2 = box_x1 + tw + 40
            box_y2 = box_y1 + th + 40

            cv2.rectangle(imgOutput, (box_x1, box_y1), (box_x2, box_y2),
                          (255, 0, 255), cv2.FILLED)

            cv2.putText(imgOutput, text, (box_x1 + 20, box_y2 - 15),
                        font, fontScale, (0, 0, 0), 7, cv2.LINE_AA)

            cv2.putText(imgOutput, text, (box_x1 + 20, box_y2 - 15),
                        font, fontScale, (255, 255, 255), 4, cv2.LINE_AA)

            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset),
                          (255, 0, 255), 3)

            
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    # EXIT the webcame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
