# Code by Adrian Rosebrock
# Modified by Franziska Mack for Parsons Summer Python Class 2020
# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

# MobileNet-SSD detection network (caffe implementation)
# https://github.com/chuanqi305/MobileNet-SSD

from imutils.video import VideoStream, FPS
import numpy as np
import imutils
import time
import cv2

# load trained model and text description of the network architecture (prototxt file)
prototxt = "/home/pi/python/Week_09/real-time-object-detection/MobileNetSSD_deploy.prototxt.txt"
model = "/home/pi/python/Week_09/real-time-object-detection/MobileNetSSD_deploy.caffemodel"
# use opencv's Deep Neural Network module to read the model in
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
# generate a set of bounding box colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize the video stream, allow the cammera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream and resize it
    # to a width of 800 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions
    h = frame.shape[0]
    w = frame.shape[1]
    # convert the frame to a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    netOutput = net.forward()

    # loop over the detections
    for detection in netOutput[0,0,:,:]:
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = float(detection[2])

        # filter out weak detections by ensuring the 'confidence' is
        # greater than 80%
        if confidence > 0.8:
            # extract the index of the class label from the 'detection'
            idx = int(detection[1])
            
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            left = int(detection[3] * w)
            top = int(detection[4] * h)
            right = int(detection[5] * w)
            bottom = int(detection[6] * h)
 
            #draw a rectangle around detected objects
            cv2.rectangle(frame, (left, top), (right, bottom),
                COLORS[idx], thickness=2)

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, label, (left, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", frame)

    # if the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()