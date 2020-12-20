import cv2
import numpy as np

class EnvPreprocessor :
  def __init__(self , preprocessing_preset):
    self.preprocessing_preset = preprocessing_preset

  def process_img(self, img):
    if self.preprocessing_preset == 1 :
      #grayscale img
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      #and resize to 84,84, color values 0-1
      return cv2.resize(img, (84, 84))[None, :, :] / 255.




