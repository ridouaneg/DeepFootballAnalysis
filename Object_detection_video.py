# IMPORTS
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from ressources import label_map_util

MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'videos/filmrole3.avi'
CWD_PATH = os.getcwd()
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'ressources','labelmap.pbtxt')
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# VIDEO CAPTURE
video = cv2.VideoCapture(PATH_TO_VIDEO)

while(True):

    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    n,p,q = np.shape(frame)
    o,l,m = np.shape(boxes)
    for i in range(l):
        if(scores[0][i] > 0.50 and classes[0][i] == 1):
            ymin, xmin, ymax, xmax = int(boxes[0][i][0] * n), int(boxes[0][i][1] * p), int(boxes[0][i][2] * n), int(boxes[0][i][3] * p)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 960, 540)
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
