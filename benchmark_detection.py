# Compare detection methods thanks to metrics and a dev set
import utils

path_image = "images/fifa_test_frame100.jpg"
img = read_image(path_image)

path_label = "labels/fifa_test_frame100.xml"
bounding_boxes = read_label(path_label)
