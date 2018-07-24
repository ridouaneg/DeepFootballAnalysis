
from utils import *
import glob
from lxml import etree
import cv2
import shutil
import numpy as np

# enter or hardcode the path of the directory
directory = str(input("Enter the directory path to the images + xml files to be augmented:"))

# get paths of xml and images files
image_path_list = glob.glob( directory + "/*.jpg")
xml_path_list = glob.glob( directory + "/*.xml")

# list of all the images in numpy format
numpy_images = []
for image_path in image_path_list:
    numpy_images.append(read_image(image_path))

numpy_images = np.array(numpy_images)


# list of all the bounding boxes coordinates (list of a list, like [[type, xmin, ymin, xmax, ymax], ...]   )
bb_coordinates = []
for xml_path in xml_path_list:
    bb_coordinates.append(read_label(xml_path))

dir_len = len(directory)


for k in range(2):

    print("k atteint")

    for i in range (len(numpy_images)):
        bounding_boxes = bb_coordinates[i]

        name = image_path_list[i][dir_len+1:-4] + "_cropped" + str(k+1) + ".jpg"

        #name2 =  image_path_list[i][dir_len+1:-4] + "_cropped2.jpg"

        #shutil.copy2(xml_path_list[i], xml_path_list[i][:-4] + "_cropped" + str(k+1) + ".xml")

        file = etree.parse(xml_path_list[i])
        root = file.getroot()

        for filename in file.xpath('/annotation/filename'):
            filename.text = str(filename.text)[:-4] + "_cropped" + str(k+1) + ".jpg"

        for path in file.xpath('/annotation/path'):
            path.text = str(path.text)[:-4] + "_cropped" + str(k+1) + ".jpg"

        for width_ in file.xpath('/annotation/size/width'):
            width = int(width_.text)

            width_.text = str(width/2)

        if( len(bounding_boxes) == len(file.xpath('/annotation/object'))):

            for noeuds in (file.xpath('/annotation/object')):

                for coord in noeuds.xpath('bndbox/xmin'):

                    if (k == 0):
                        if (int(coord.text) > width /2):
                            root.remove(noeuds)

                    else:
                        if (int(coord.text) < width/2):
                            root.remove(noeuds)
                        else:
                            coord.text = str(int(coord.text)- width/2)

                for coord in noeuds.xpath('bndbox/xmax'):

                    if (k == 0):
                        if (int(coord.text) > width /2):
                        
                                root.remove(noeuds)

                    if(k == 1):
                        if (int(coord.text) > width/2):
                            coord.text = str(int(coord.text)- width/2)

        file.write(xml_path_list[i][dir_len+1:-4] + "_cropped" + str(k+1) + ".xml") # to save xml with rights labels
        cv2.imwrite(name,numpy_images[i][0:numpy_images[i].shape[0] , k*numpy_images[i].shape[1]/2:(k+1)*numpy_images[i].shape[1]/2])
