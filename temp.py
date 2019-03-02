import requests
from PIL import Image
import numpy as np
import base64
import json

from detect_faces import detect_faces

cordinates, confidence = detect_faces(prototxt="deploy.prototxt.txt",image="9ppl.jpg",model="res10_300x300_ssd_iter_140000.caffemodel",confidence=0.7)

print ("DONE WITH EXECUTION!!!")
print (cordinates, confidence)