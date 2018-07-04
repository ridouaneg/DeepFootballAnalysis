# Using YOLO to detect individuals and the ball in frames
#encoding : utf-8

from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

def detection_yolo(image_path):
    large, height = 1920, 1080
    div = 2
    net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True)
    x, img = data.transforms.presets.ssd.load_test(image_path, short=int(height/div)) # short_max = 560
    box_ids, scores, bboxes = net(x)
    box_ids, scores, bboxes = box_ids.asnumpy(), scores.asnumpy(), bboxes.asnumpy()

    bndbx_detected = []
    n, p, q = box_ids.shape
    for i in range(p):
        if(net.classes[int(box_ids[0][i][0])] == 'person'):
            xmin, ymin, xmax, ymax = int(bboxes[0][i][0]*div), int(bboxes[0][i][1]*div), int(bboxes[0][i][2]*div), int(bboxes[0][i][3]*div)
            proba = scores[0][i][0]
            bndbx_detected.append([(xmin, ymin, xmax, ymax), proba])

    return bndbx_detected
