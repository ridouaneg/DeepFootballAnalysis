# Compare detection methods thanks to metrics and a dev set
import utils
import numpy as np
import os
import detection_yolo as yolo
import matplotlib.pyplot as plt

# get predictions from the detection method
def predictions(image_path, detection_method="yolo"):
    bndbx_detected = []

    if detection_method=="yolo":
        bndbx_detected = yolo.detection_yolo(image_path)
    elif detection_method=="rcnn":
        bndbx_detected = yolo.detection_rcnn(image_path)
    elif detection_method=="kmeans":
        print(" ")
    else:
        print("The detection method doesn't exist.")

    return bndbx_detected

def mean(l):
    n = len(l)
    if n == 0:
        res = 0
    else:
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

def IoU(bndbx_d, bndbx_l):
    xmin1, ymin1, xmax1, ymax1 = bndbx_d
    label, xmin2, ymin2, xmax2, ymax2 = bndbx_l

    xA = max(xmin1, xmin2)
    yA = max(ymin1, ymin2)
    xB = min(xmax1, xmax2)
    yB = min(ymax1, ymax2)
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    aire1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    aire2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    union = aire1 + aire2 - intersection

    eps = 10**(-8)
    iou = intersection / (union + eps)

    return iou

# calculate precision and recall
def precision_and_recall_top_k(bndbx_truth, bndbx_detected, IoU_min, k):
    # ATTENTION : bndbx_detected = [ [(x1, y1, x2, y2), probability] , ...]
    # ATTENTION : precision and recall needs to be calculated on top K predictions
    # classed by probability
    # ATTENTION : we need to remove double detection
    bndbx_detected = bndbx_detected[0:k]
    n_T = len(bndbx_truth)
    n_P = k
    n_TP = 0
    n_FP = 0

    for i in range(n_P):
        l_tmp = [IoU(bndbx_detected[i][0], bndbx_truth[j][0]) for j in range(n_T)]
        if (max(l_tmp) >= IoU_min):
            n_TP += 1
        else:
            n_FP += 1

    precision = n_TP / n_P
    recall = n_TP / n_T

    return precision, recall

def precision_and_recall_list(bndbx_truth, bndbx_detected, IoU_min):
    precisions = []
    recalls = []

    for k in range(1, len(bndbx_detected)):
        precision, recall = precision_and_recall_top_k(bndbx_truth, bndbx_detected, IoU_min, k)
        precisions.append(precision)
        recalls.append(recall)

    precisions.append(0)
    recalls.append(1)

    return precisions, recalls

def AP(bndbx_truth, bndbx_detected, IoU_min):
    precisions, recalls = precision_and_recall_list(bndbx_truth, bndbx_detected, IoU_min)
    tmp = [0.1*i for i in range(11)]
    sum = 0

    try:
        i_tmp = recalls.index(1) + 1
    except ValueError:
        i_tmp = len(precisions) + 1

    precisions = precisions[:i_tmp]
    recalls = recalls[:i_tmp]

    print('Precisions | Recalls')
    print('--------------------')
    for j in range(len(precisions)):
        print("%.2f" % precisions[j], '      |   ', "%.2f" % recalls[j])

    #plt.plot(recalls, precisions)
    #plt.show()

    for r in tmp:
        # ind = AP_compl(recalls, r) # premier indice de recalls tq la valeur soit >= r
        sum += max(precisions[AP_compl(recalls, r):])

    AP = (1 / 11) * sum
    return AP

def mAP(path_test_repo_images, path_test_repo_labels, detection_method="yolo", IoU_min = 0.50):
    """
        Given two directories (images and associated labels) and a detection method,
        return de Mean Average Precision for this method on the given datas.
        Attention : datas must be sorted in alphabetical order !
    """

    image_filenames = []
    label_filenames = []
    for image_filename in os.listdir('images'):
        image_filenames.append('images' + '/' + image_filename)
    for label_filename in os.listdir('labels'):
        label_filenames.append('labels' + '/' + label_filename)
    image_filenames = sorted(image_filenames)
    label_filenames = sorted(label_filenames)

    APs = []

    print('')
    print('mAP evaluation on the images of the test repository "images"')
    print('Configuration : IoU >= ', IoU_min)

    for image_filename, label_filename in zip(image_filenames, label_filenames):

        print('')
        print('Image name : ' + image_filename)
        print('Processing..........')

        bndbx_truth = utils.read_label(label_filename)
        bndbx_detected = predictions(image_filename, detection_method)
        print('Number of labelled objects : ', len(bndbx_truth))
        print('Number of detected objects : ', len(bndbx_detected))

        ap = AP(bndbx_truth, bndbx_detected, IoU_min)
        APs.append(ap)
        print('Average Precision (AP) : ', ap)

        print('..........Finished')

    mAP = mean(APs)
    print('')
    print('Mean Average Precision (mAP) : ', mAP)
    print('')

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
