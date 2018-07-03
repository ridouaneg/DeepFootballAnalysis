# Compare detection methods thanks to metrics and a dev set
import utils
import numpy as np
#import detection

# get predictions from the detection method
def predictions(path_image, detection_method="yolo"):
    bndbx_detected = []

    if detection_method=="yolo":
        #bndbx_detected = detection.yolo(path_image)
        print(" ")
    elif detection_method=="tf":
        print(" ")
    elif detection_method=="opencv":
        print(" ")
    else:
        print("The detection method doesn't exist.")

    return bndbx_detected

def IoU(bndbx1, bndbx2):
    label, xmin1, ymin1, xmax1, ymax1 = bndbx1
    label, xmin2, ymin2, xmax2, ymax2 = bndbx2

    intersection = 0
    if xmax1 >= xmin2 and xmax1 <= xmax2:
        if ymax2 >= ymin1 and ymax2 <= ymax1:
            intersection = (xmax1 - xmin2) * (ymax2 - ymin1)
    aire1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    aire2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = aire1 + aire2 - intersection

    eps = 10**(-8)
    iou = intersection / (union + eps)

    return iou

# calculate precision and recall
def precision_and_recall(bndbx_truth, bndbx_detected, IoU_min):
    # ATTENTION : bndbx_detected = [(label, x1, y1, x2, y2, probability), ...]
    # ATTENTION : precision and recall needs to be calculated on top K predictions
    # classed by probability
    # ATTENTION : we need to remove double detection
    n_T = len(bndbx_truth)
    n_P = len(bndbx_detected)
    n_TP = 0
    n_FP = 0

    for i in range(n_P):
        if (max([IoU(bndbx_detected[i], bndbx_truth[j]) for j in range(n_T)]) >= IoU_min):
            n_TP += 1
        else:
            n_FP += 1

    precision = n_TP / (n_P)
    recall = n_TP / n_T

    return precision, recall

# given a frame and its labels :
# path_image, path_label
path_image = "images/fifa_test_frame100.jpg"
path_label = "labels/fifa_test_frame100.xml"
img = utils.read_image(path_image)
bndbx_truth = utils.read_label(path_label)
#bndbx_detected = predictions(path_image)
bndbx_detected = bndbx_truth
precision, recall = precision_and_recall(bndbx_truth, bndbx_detected, 0.50)
print(precision, recall)
