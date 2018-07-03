# Using YOLO to detect individuals and the ball in frames
#encoding : utf-8

#opening the file containing what yolov3 returns after processing an image
# you need to run the programm in the folder containing prediction_details.txt
file = open("prediction_details.txt",'r')

#tuple containing our data in the right format
data = ()

try:
    for line in file:
        # processing the line, removing spaces
        buffer = [ i for i in line.split(" ") if i != '']
        #format of an element of the tuple returned :
        # (class, left, top, right, bottom, probability) (all of them being strings)
        data.append((buffer[1],buffer[4],buffer[5],buffer[6],buffer[7],buffer[2]))

finally:
    file.close()
    print(data)
