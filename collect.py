import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import string

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300


# ALWAYS LOAD CHART =====
base_path = os.path.dirname(os.path.abspath(__file__))
chart_path = os.path.join(base_path,"HAND SIGN REFERENCE.jpg")
chart = cv2.imread(chart_path)

if chart is None:
    print(f"⚠ chart NOT FOUND at: {chart_path}")
else:
    chart = cv2.resize(chart, (500, 500))
    cv2.imshow("ASL CHART", chart)

# Default class
current_label = "A"
base_folder = "Data"
counter = 0

# Create 26 folders automatically
for letter in string.ascii_uppercase:
    os.makedirs(os.path.join(base_folder, letter), exist_ok=True)

print("Press A–Z to select class")
print("Press SPACE to save image")
print("Press Q to quit")

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size != 0:

            cropH, cropW, _ = imgCrop.shape
            aspectRatio = cropH / cropW

            #SAFE RESIZING
            if aspectRatio > 1:
                scale = imgSize / cropH
                newW = int(cropW * scale)
                imgResize = cv2.resize(imgCrop, (newW, imgSize))
                wGap = (imgSize - newW) // 2
                imgWhite[:, wGap:wGap + newW] = imgResize

            else:
                scale = imgSize / cropW
                newH = int(cropH * scale)
                imgResize = cv2.resize(imgCrop, (imgSize, newH))
                hGap = (imgSize - newH) // 2
                imgWhite[hGap:hGap + newH, :] = imgResize

            
            cv2.imshow("ImageWhite", imgWhite)

        #Show current class name
        cv2.putText(imgOutput, f"Class: {current_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    #Save image
    if key == 32:  # SPACE
        counter += 1
        save_path = os.path.join(base_folder, current_label,
                                 f"Image_{time.time()}.jpg")
        cv2.imwrite(save_path, imgWhite)
        print(f"Saved {counter} images in {current_label}")

    #Switch class (A–Z)
    if key in range(65, 91):
        current_label = chr(key)
        counter = 0
        print(f"Switched to class: {current_label}")

    #Quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
