import os
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
plant_species = ["Black-grass","Charlock","Cleavers","Common Chickweed","Common wheat","Fat Hen","Loose Silky-bent","Maize","Scentless Mayweed","Shepherds Purse","Small-flowered Cranesbill","Sugar beet"]

print("enter the absolute path of the image \n")
img_path = input()
print("pick a model : custom cnn model for raw data,custom cnn model for processed data, inception model")
model_name = input()
model = tf.keras.models.load_model('/content/drive/MyDrive/Data/Plant seedling classification/saved_models'+model)
# Check its architecture
model.summary()

def load_img(path):
  img = cv2.imread(path,cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  resized = cv2.resize(img,(100,100), interpolation = cv2.INTER_AREA)
  return resized

def process_img(img):
  img1 = tfa.image.gaussian_filter2d(img,filter_shape=(4,4),sigma=100)
  img2 = cv2.cvtColor(np.float32(img1),cv2.COLOR_RGB2HSV)
  green_segment_values = 50
  hsv_lower_bound = np.array([100-green_segment_values,0,0])
  hsv_upper_bound = np.array([100+green_segment_values,255,240])
  mask = cv2.inRange(img2,hsv_lower_bound,hsv_upper_bound)
  kernel = np.ones((3,3),np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  return mask


def make_prediction(model,img):
   output = model.predict(img)
   specie_index = np.argmax(output, axis=1)
   return plant_species[specie_index]


loaded_img = load_img(img_path)

if model_name == "custom cnn model for processed data":
  loaded_img = process_img(loaded_img)

print(make_prediction(model,loaded_img))
