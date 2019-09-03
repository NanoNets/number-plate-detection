import os, sys
from PIL import Image
import cv2
import pandas as pd

# The annotation file consists of image names, text label, 
# bounding box information like xmin, ymin, xmax and ymax.
ANNOTATION_FILE = 'data/annot_file.csv'
df = pd.read_csv(ANNOTATION_FILE)

#image directory path
IMG_DIR = 'data/images'
# The cropped images will be stored here
CROP_DIR = 'data/crops'

files = df['files']

size = (200,200)

for file in files:
	print(file)
	img = cv2.imread(IMG_DIR +'/' + file)
	annot_data = df[df['files'] == file]
	xmin = int(annot_data['xmin'])
	ymin = int(annot_data['ymin'])
	xmax = int(annot_data['xmax'])
	ymax = int(annot_data['ymax'])
	crop = img[ymin:ymax,xmin:xmax]
	new_crop = cv2.resize(crop, dsize=size, interpolation=cv2.INTER_CUBIC)
	new_crop.save(CROP_DIR + '/' + file.split('.')[0] + '.png', 'PNG', quality=90)
	