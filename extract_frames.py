# encoding: utf-8
# Extraire les frames des vidéos présentes dans le dossier ressources
# et les enregistrer au format .jpg dans un sous-dossier de ressources
# dont le nom est nom_de_la_video_frames, les frames ayant les noms suivants
# nom_de_la_video_frame1, nom_de_la_video_frame2, ...

# Après labellisation, enregistrer les labels au chemin DeepFootballAnalysis/
# ressources/nom_de_la_video_labels/


import os
import cv2
import numpy as np

#get the absolute path of the video used
path_video = str(input("entree nom video: "))
path =  path_video + "_frames"

#create the directory where we'll store the frames extracted
if not os.path.exists(path):
    os.makedirs(path)

vidcap = cv2.VideoCapture(path_video)
success,image = vidcap.read()
count = 0
success = True
fps = vidcap.get(cv2.CAP_PROP_FPS)
lengthframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

print "fps = ", fps
print "Video de duree : ", lengthframe/fps
intervalle = int(input("enter the number of frames between each frame extracted: "))
#choose when we want to extract frames
debut = 1
fin = int(lengthframe/fps)


while success:

  if (count > debut*fps) & (count % intervalle ==0) & (count < fin*fps):
    cv2.imwrite(os.path.join(path , "frame%d.jpg" % count), image)

  success,image = vidcap.read()
  count += 1
