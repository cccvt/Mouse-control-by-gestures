import pyautogui
import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class MouseControl:
    def __str__(self):
        description = "\tMouse control by hand's gestures\n"
        return description

    def __init__(self, x_start=None, y_start=None, x_finish=None, y_finish=None):
        # Mouse position
        self.x_start, self.y_start = x_start, y_start
        self.x_finish, self.y_finish = x_finish, y_finish
        # Error if the mouse cursor is in the upper left corner of the screen.
        pyautogui.FAILSAFE = False
        # Was there a palm in the image?
        self.palm_was = False

    @staticmethod
    def move_to(x, y, duration=0.6):
        """ Move cursor to position """
        pyautogui.moveTo(x, y, duration=duration)

    def cursor_coordinates(self, box, height, width):
        y_min, x_min, y_max, x_max = box[0], box[1], box[2], box[3]
        x, y = (1 - (x_min + (x_max - x_min) / 2)) * width // 1, (y_min + (y_max - y_min) / 2) * height // 1
        return x, y

    def action(self, gesture, x, y, dist_div=1.7):
        """ Mouse action """
        if gesture == 'Empty':
            pass
        elif gesture == 'palm':
            # Get cursor position
            self.x_start, self.y_start = x, y
            self.x_finish, self.y_finish = None, None
            self.palm_was = True
        elif gesture == 'fist' and self.palm_was:
            # Move cursor
            self.x_finish, self.y_finish = x, y
            x_mouse, y_mouse = pyautogui.position()
            x_dest = x_mouse + (self.x_finish - self.x_start)//dist_div
            y_dest = y_mouse + (self.y_finish - self.y_start)//dist_div
            self.move_to(x_dest, y_dest, 0.2)
        elif gesture == '1finger' and self.palm_was:

            # Left button click
            pyautogui.click(pyautogui.position())
        elif gesture == '3fingers' and self.palm_was:

            # Right button click
            pyautogui.rightClick(pyautogui.position())
        elif gesture == '2fingers' and self.palm_was:

            # Left button double click
            pyautogui.click(pyautogui.position())
            pyautogui.click(pyautogui.position())


class VideoStream:
    def __str__(self):
        description = '\tGet image from webcamera\n'
        return description

    def __init__(self, width=480, height=360, camera_num=0):
        # Initialize the webcamera stream
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(camera_num)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.RUN = True

    def get_img(self):
        """ Get image from web camera """
        ret, img = self.stream.read()

        return ret, img


    def stop(self):
        """ Stop video stream """
        self.stream.release()

class HandDetector:
    def __str__(self):
        description = "\tDetect Hand on image\n"
        return description

    def __init__(self, path_to_model='./data/faster_rcnn.pb',
                 path_to_label_map='./data/label_map.pbtxt',
                 num_classes=5):
        """ Load model, label map and initialize TF session """
        label_map = label_map_util.load_labelmap(path_to_label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

    def detect_objects(self, image_np, num_boxes=1):
        """ Detect hand on image """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        (class_name, box) = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4, max_boxes_to_draw=num_boxes)
        # Return image, gesture's class and bounding box
        return image_np, class_name, box

if __name__ == '__main__':
    print("""There are classes for mouse control by hand's gestures""")