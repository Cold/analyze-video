#!/usr/bin/env python3
import numpy as np
import argparse
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
# convert RGB to HSV
r = 232
g = 255
b = 66
color = np.uint8([[[b, g, r]]])
rgb_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
print(rgb_hsv[0][0][0]-10, 100, 100)
# greenLower = (22, 57, 191)
# greenUpper = (76, 193, 255)

# greenLower = (29, 60, 136)## pop.mp4 colors
# greenUpper = (35, 181, 255)

# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)

if not args.get("video", False):
    camera = cv2.VideoCapture(8)
else:
    camera = cv2.VideoCapture(args["video"])
# fps = cv2.VideoCapture.get(cv2.CAP_PROP_FPS)

while True:
    (grabbed, frame) = camera.read()
    time.sleep(.1)
    if args.get("video") and not grabbed:
        break
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit = ((rgb_hsv[0][0][0])-10, 100, 100)
    upperLimit = ((rgb_hsv[0][0][0])+10, 255, 255)
    kernel = np.zeros((2, 2), np.uint8)
    mask = cv2.inRange(hsv, lowerLimit, upperLimit)
    # mask = cv2.erode(mask, kernel)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 10:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius < 20:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
