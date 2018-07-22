from utils import *
import glob
import imgaug as ia
from imgaug import augmenters as iaa
from lxml import etree
import cv2
import shutil
import numpy as np

# flip images horizontally : aug = iaa.Fliplr(0.5)
# enter or hardcode the path of the directory
directory = str(input("Enter the directory path to the images + xml files to be augmented:"))
dir_len = len(directory)
# get paths of xml and images files
image_path_list = glob.glob( directory + "/*.jpg")
xml_path_list = glob.glob( directory + "/*.xml")

'''
print(image_path_list)
print(xml_path_list)
'''

# list of all the images in numpy format
numpy_images = []
for image_path in image_path_list:
    numpy_images.append(read_image(image_path))

# list of all the bounding boxes coordinates (list of a list, like [[type, xmin, ymin, xmax, ymax], ...]   )
bb_coordinates = []
for xml_path in xml_path_list:
    bb_coordinates.append(read_label(xml_path))




if ( len(numpy_images) == len(image_path_list)):

    for i in range (len(numpy_images)):
        # width = np.array(numpy_images[i]).shape[1]


        bounding_boxes = bb_coordinates[i]  # list of all the bounding boxes of the i-th image
                                            # of shape [[('type',xmin, ymin, xmax, ymax)], ... , ]



        name = image_path_list[i][dir_len+1:-4] + "_flipped.jpg"

        # shutil.copy2(xml_path_list[i], xml_path_list[i][:-4] + "_flipped.xml")
        file = etree.parse(xml_path_list[i])

        for filename in file.xpath('/annotation/filename'):
            filename.text = str(filename.text )[:-4] + "_flipped.jpg"

        for path in file.xpath('/annotation/path'):
            path.text = str(path.text)[:-4] + "_flipped.jpg"

        for width_ in file.xpath('/annotation/size/width'):
            width = int(width_.text)

        if( len(bounding_boxes) == len(file.xpath('/annotation/object'))):

            for noeuds in range (len(file.xpath('/annotation/object'))):

                for coord in file.xpath('/annotation/object')[noeuds].xpath('bndbox/xmin'):

                    xm = int(coord.text)

                    xm = int(width) - int(bounding_boxes[noeuds][0][3])
                    coord.text = str(int(xm))

#   shape [[('type',xmin, ymin, xmax, ymax)], ... , ]

                for coord in file.xpath('/annotation/object')[noeuds].xpath('bndbox/xmax'):
                    xM = int(coord.text)

                    xM = int(width) - int(bounding_boxes[noeuds][0][1])

                    coord.text = str(int(xM))


        file.write(xml_path_list[i][dir_len+1:-4] + "_flipped.xml") # to save xml with rights labels
        cv2.imwrite(name, cv2.flip(numpy_images[i],1))     # to save flipped images
