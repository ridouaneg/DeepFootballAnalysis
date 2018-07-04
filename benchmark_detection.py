# Compare detection methods thanks to metrics and a dev set
import utils
import numpy as np
import os
#import detection

# get predictions from the detection method
def predictions(image, detection_method="yolo"):
    bndbx_detected = []

    if detection_method=="yolo":
        #bndbx_detected = detection.yolo(path_image)
        print(" ")
    elif detection_method=="hsv":
        print(" ")
    elif detection_method=="kmeans":
        print(" ")
    else:
        print("The detection method doesn't exist.")

    return bndbx_detected

def mean(l):
    if n == 0:
        res = 0
    else:
        n = len(l)
        sum = 0
        for i in range(n):
            sum += l[i]
        res = sum / n
    return res

def AP_compl(recalls, r):
    # retourne le premier indice de recalls tq la valeur soit >= r
    ind = 0
    while recalls[ind] < r:
        ind += 1
    return ind

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
def precision_and_recall_top_k(bndbx_truth, bndbx_detected, IoU_min, k):
    # ATTENTION : bndbx_detected = [ [(label, x1, y1, x2, y2), probability] , ...]
    # ATTENTION : precision and recall needs to be calculated on top K predictions
    # classed by probability
    # ATTENTION : we need to remove double detection
    bndbx_detected = bndbx_detected[0:k]
    n_T = len(bndbx_truth)
    n_P = len(bndbx_detected)
    n_TP = 0
    n_FP = 0

    for i in range(n_P):
        if (max([IoU(bndbx_detected[i][0], bndbx_truth[j][0]) for j in range(n_T)]) >= IoU_min):
            n_TP += 1
        else:
            n_FP += 1

    precision = n_TP / (n_P)
    recall = n_TP / n_T

    return precision, recall

def precision_and_recall_list(bndbx_truth, bndbx_detected, IoU_min):
    precisions = []
    recalls = []

    for k in range(len(bndbx_detected)):
        precision, recall = precision_and_recall_top_k(bndbx_truth, bndbx_detected, IoU_min, k)
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def AP(bndbx_truth, bndbx_detected, IoU_min):
    precisions, recalls = precision_and_recall_list(bndbx_truth, bndbx_detected, IoU_min)
    tmp = [0.10*i for i in range(11)]
    for r in tmp:
        ind = AP_compl(recalls, r) # premier indice de recalls tq la valeur soit >= r
        l = precisions[ind:]
        p = max(l)
        sum += p
        # sum += max(precisions[AP_compl(recalls, r):])
    AP = (1 / 11) * sum
    return AP

def mAP(path_test_repo_images, path_test_repo_labels):
    IoU_min = 0.50
    APs = []

    image_filenames = []
    label_filenames = []
    for image_filename in os.listdir('images'):
        image_filenames.append('images' + '/' + image_filename)
    for label_filename in os.listdir('labels'):
        label_filenames.append('labels' + '/' + label_filename)
    image_filenames = sorted(image_filenames)
    label_filenames = sorted(label_filenames)

    for image_filename, label_filename in image_filenames, label_filenames:
        bndbx_truth = read_label(label_filename)
        bndbx_detected = predictions(read_image(image_filename), detection_method="yolo")
        APs.append(AP(bndbx_truth, bndbx_detected, IoU_min))
    mAP = mean(APs)

    return mAP

# given a frame and its labels :
# path_image, path_label
#path_image = "images/fifa_test_frame100.jpg"
#path_label = "labels/fifa_test_frame100.xml"
#img = utils.read_image(path_image)
#bndbx_truth = utils.read_label(path_label)
#bndbx_detected = predictions(path_image)
#bndbx_detected = bndbx_truth
#precision, recall = precision_and_recall(bndbx_truth, bndbx_detected, 0.50)
#print(precision, recall)
