from utils import *
import cv2
from time import time


detector = HandDetector()
stream = VideoStream(camera_num=0)
mouse = MouseControl()

if __name__ == '__main__':
    RUN = True
    start_time = time()
    while RUN:
        ret, img = stream.get_img()
        boxed_image, class_name, box = detector.detect_objects(img)
        if len(box):
            x, y = mouse.cursor_coordinates(box, stream.height, stream.width)
            mouse.action(class_name, x, y)
        current_time = time()
        fps = 10.0/(current_time-start_time)
        cv2.putText(boxed_image, "FPS = {}".format(fps), (20, 20), 0, 0.5, (0, 0, 255))
        cv2.imshow('Image', boxed_image)
        if cv2.waitKey(10) & 0xFF == 27:
            RUN = not RUN


