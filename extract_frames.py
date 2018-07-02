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


path = os.getcwd() + "/ressources/fifa_test"

if not os.path.exists(path):
    os.makedirs(path)

vidcap = cv2.VideoCapture('./ressources/fifa_test.flv')
success,image = vidcap.read()
count = 0
success = True
fps = vidcap.get(cv2.CAP_PROP_FPS)
lengthframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print "fps = ", fps
print "Video de duree : ", lengthframe/fps
debut = input("Saisissez le début de la sauvegarde des images (s)")
fin = input("Saisissez la fin de la sauvegarde des images (s)")


while success:

  if (count > debut*fps) & (count % 25 ==0) & (count < fin*fps):
    cv2.imwrite(os.path.join(path , "frame%d.jpg" % count), image)
    print 'saved a new frame: n° ', success
  success,image = vidcap.read()
  count += 1
