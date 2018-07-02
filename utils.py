import cv2
from lxml import etree

def read_image(path_image):
    """
    Input :
        path - path (String) to the image

    Output :
        img - the image (numpy array)
    """
    img = cv2.imread(path_image)
    return img

def read_label(path_label):
    """
    Input :
        path - path (String) to the label

    Output :
        bounding_boxes - labels (list)
            bounding_boxes = [('label', xmin, ymin, xmax, ymax), ...]
    """
    file = etree.parse(path_label)
    bounding_boxes = []
    for noeuds in file.xpath('/annotation/object'):
        for name in noeuds.xpath('name'):
            type = name.text
        for coord in noeuds.xpath('bndbox/xmin'):
            xm = int(coord.text)
        for coord in noeuds.xpath('bndbox/ymin'):
            ym = int(coord.text)
        for coord in noeuds.xpath('bndbox/xmax'):
            xM = int(coord.text)
        for coord in noeuds.xpath('bndbox/ymax'):
            yM = int(coord.text)
        bounding_boxes.append((type, xm, ym, xM, yM))
    return bounding_boxes
