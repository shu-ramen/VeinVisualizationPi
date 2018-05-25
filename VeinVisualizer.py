# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
import RPi.GPIO as gpio
import numpy as np
import time
import math
import cv2
import sys
import os

class VeinVisualizer:
  BAR_PWM = 0
  BAR_LUT_A = 1
  BAR_LUT_B = 2
  WINDOW_NAME = "Vein Visualizer"
  VALUE_BAR_NAME = "value"
  SELECT_BAR_NAME = "0:duty  1:contrast  2:threshold"

  def __init__(self, frameWidth, frameHeight, framerate, pwmPort, pwmFrequency):
    # PWM
    self.pwmPort = pwmPort
    self.pwmFrequency = pwmFrequency
    self.pwm = 0

    # camera
    self.frameWidth = frameWidth
    self.frameHeight = frameHeight
    self.framerate = framerate
    self.camera = 0
    self.rawCapture = 0

    # trackBar
    self.barSelect = VeinVisualizer.BAR_PWM
    self.duty_val = 100
    self.lut_a_val = 128
    self.lut_b_val = 128
    self.lut = np.zeros(256)
    self.__updateLut(self.lut_a_val, self.lut_b_val)

  # setup
  def setup(self):
    self.__setupCamera()
    self.__setupGpio()
    self.__setupWindow()

  # initialize the camera and grab a reference to the raw camera capture
  def __setupCamera(self):
    self.camera = PiCamera()
    self.camera.resolution = (self.frameWidth, self.frameHeight)
    self.camera.framerate = self.framerate
    self.rawCapture = PiRGBArray(self.camera, size=(self.frameWidth, self.frameHeight))
    time.sleep(0.1) # camera warmup

  # initialize gpio
  def __setupGpio(self):
    gpio.setwarnings(False)
    gpio.setmode(gpio.BOARD)
    gpio.setup(self.pwmPort, gpio.OUT)
    self.pwm = gpio.PWM(self.pwmPort, self.pwmFrequency)
    self.pwm.start(1)

  # initialize window
  def __setupWindow(self):
    cv2.namedWindow(VeinVisualizer.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(VeinVisualizer.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.createTrackbar(VeinVisualizer.VALUE_BAR_NAME, VeinVisualizer.WINDOW_NAME, 0, 255, self.__updateBar)
    cv2.createTrackbar(VeinVisualizer.SELECT_BAR_NAME, VeinVisualizer.WINDOW_NAME, 0, 2, self.__changeBar)
    cv2.setTrackbarPos(VeinVisualizer.VALUE_BAR_NAME, VeinVisualizer.WINDOW_NAME, self.duty_val)
    cv2.setTrackbarPos(VeinVisualizer.SELECT_BAR_NAME, VeinVisualizer.WINDOW_NAME, VeinVisualizer.BAR_PWM)

  # updateBar
  def __updateBar(self, val):
    if (self.barSelect == VeinVisualizer.BAR_PWM):
      self.duty_val = val
      self.pwm.ChangeDutyCycle(self.duty_val/8)  # PWM LIMITATION
    if (self.barSelect == VeinVisualizer.BAR_LUT_A):
      self.lut_a_val = val
      self.__updateLut(self.lut_a_val, self.lut_b_val)
    if (self.barSelect == VeinVisualizer.BAR_LUT_B):
      self.lut_b_val = val
      self.__updateLut(self.lut_a_val, self.lut_b_val)
  
  # changeBar
  def __changeBar(self, val):
    self.barSelect = val
    if (self.barSelect == VeinVisualizer.BAR_PWM):
      cv2.setTrackbarPos(VeinVisualizer.VALUE_BAR_NAME, VeinVisualizer.WINDOW_NAME, self.duty_val)
    if (self.barSelect == VeinVisualizer.BAR_LUT_A):
      cv2.setTrackbarPos(VeinVisualizer.VALUE_BAR_NAME, VeinVisualizer.WINDOW_NAME, self.lut_a_val)
    if (self.barSelect == VeinVisualizer.BAR_LUT_B):
      cv2.setTrackbarPos(VeinVisualizer.VALUE_BAR_NAME, VeinVisualizer.WINDOW_NAME, self.lut_b_val)

  # update lut
  def __updateLut(self, a, b):
    self.lut = [ np.uint8(255.0 / (1 + math.exp(-(a/5) * (i - b) / 255.))) for i in range(256)]

  # create contrast image
  def __contrast(self, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resultImg = np.array( [ self.lut[value] for value in gray.flat], dtype=np.uint8 )
    resultImg = resultImg.reshape(gray.shape)
    return resultImg

  # get hand mask
  def __getHandMaskAndOutline(self, contrast):
    ret, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get maximum size convex
    maxIdx = -1
    maxLen = -1
    for i in range(len(contours)):
      if (maxLen < len(contours[i])):
        maxIdx = i
        maxLen = len(contours[i])

    # Create Mask
    mask = np.zeros(contrast.shape, dtype=np.uint8)
    mask.fill(0)
    outline = -1
    if maxIdx > 0:
      outline = [ contours[maxIdx] ]
      cv2.fillPoly(mask, outline, (255))

    return mask, outline

  
  def __getVeinMask(self, contrast):
    veinMask = cv2.Canny(contrast, self.lut_b_val-40, self.lut_b_val+5)
    kernel = np.ones((3, 3), np.uint8)
    veinMask = cv2.dilate(veinMask, kernel)
    return veinMask


#    row = 4
#    col = 3
#    w = int(self.frameWidth / row)
#    h = int(self.frameHeight / col)
#    clips = [[]]

#    for y in range(col):
#      clips.append([])
#      for x in range(row):
#        rect = contrast[y*h:(y+1)*h, x*w:(x+1)*w]
#        ret, rectBw = cv2.threshold(rect, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#        rectBw = cv2.bitwise_not(rectBw)
#        rectBwRGB = cv2.cvtColor(rectBw, cv2.COLOR_GRAY2RGB)
#        clips[y].append(rectBw)
#    
#    veinMask = cv2.vconcat([ cv2.hconcat([clips[y][x] for x in range(row)]) for y in range(col) ])
#
#    return veinMask

  # getVein
  def __getVein(self, handMask, outline, veinMask, original):
    if outline == -1:
      return original

    red = np.zeros(original.shape, np.uint8)
    red[:] = (0, 0, 255)
#    mask = cv2.bitwise_and(handMask, handMask, mask=veinMask)
#    kernel = np.ones((5, 5), np.uint8)
#    mask = cv2.dilate(mask, kernel)
    vein = cv2.bitwise_and(red, red, mask=handMask)
    vein = cv2.bitwise_and(vein, vein, mask=veinMask)
    vein = cv2.bitwise_or(original, vein)

    cv2.drawContours(vein, outline, 0, (0, 255, 255))

    return vein
  
  # visualize
  def visualize(self, img_original):
    img_contrast = self.__contrast(img_original)
    img_handMask, outline = self.__getHandMaskAndOutline(img_contrast)
    img_veinMask = self.__getVeinMask(img_contrast)
    img_result = self.__getVein(img_handMask, outline, img_veinMask, img_original)

    return img_contrast, img_handMask, img_veinMask, img_result
  
  # KEY PRESS ACTION
  def __keyAction(self, key, img_original, img_contrast, img_handMask, img_veinMask, img_result):
      # Change TrackBar Setting
      if key == ord(str(VeinVisualizer.BAR_PWM)):
        cv2.setTrackbarPos(VeinVisualizer.SELECT_BAR_NAME, VeinVisualizer.WINDOW_NAME, VeinVisualizer.BAR_PWM)
      if key == ord(str(VeinVisualizer.BAR_LUT_A)):
        cv2.setTrackbarPos(VeinVisualizer.SELECT_BAR_NAME, VeinVisualizer.WINDOW_NAME, VeinVisualizer.BAR_LUT_A)
      if key == ord(str(VeinVisualizer.BAR_LUT_B)):
        cv2.setTrackbarPos(VeinVisualizer.SELECT_BAR_NAME, VeinVisualizer.WINDOW_NAME, VeinVisualizer.BAR_LUT_B)
      
      # Save Imagess
      if key == ord("s"):
        now = datetime.now().strftime('%Y%m%d/%Hh%Mm%Ss')
        os.makedirs("../data/{0}".format(now))
        cv2.imwrite("../data/{0}/original.jpg".format(now), img_original)
        cv2.imwrite("../data/{0}/contrast.jpg".format(now), img_contrast)
        cv2.imwrite("../data/{0}/handMask.jpg".format(now), img_handMask)
        cv2.imwrite("../data/{0}/veinMask.jpg".format(now), img_veinMask)
        cv2.imwrite("../data/{0}/outbut.jpg".format(now), img_result)

        with open("../data/{0}/parameter.json".format(now), mode="w") as f:
          f.write("{\n")
          f.write("    \"duty_val\" : {0:d},\n".format(self.duty_val))
          f.write("    \"lut_a_val\" : {0:d},\n".format(self.lut_a_val))
          f.write("    \"lut_b_val\" : {0:d},\n".format(self.lut_b_val))
          f.write("    \"thresh_val\" : {0:d}\n".format(self.lut_b_val))
          f.write("}")

  def streaming(self):
    for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
      # grab the raw NumPy array representing the image, then initialize the timestamp
      # and occupied/unoccupied text
      img_original = frame.array
      img_contrast, img_handMask, img_veinMask, img_result = self.visualize(img_original)

      # show the Vein Visualizer
      img_to_show = cv2.hconcat([cv2.cvtColor(img_contrast, cv2.COLOR_GRAY2RGB), cv2.cvtColor(img_veinMask, cv2.COLOR_GRAY2RGB), img_result])
      cv2.imshow(VeinVisualizer.WINDOW_NAME, img_to_show)
      key = cv2.waitKey(1) & 0xFF

      # clear the stream in preparation for the next Frame
      self.rawCapture.truncate(0)

      # Quit the program
      if key == ord("q"):
        self.pwm.stop()
        gpio.cleanup()
        break
      
      # key press action
      self.__keyAction(key, img_original, img_contrast, img_handMask, img_veinMask, img_result)

def main():
  vv = VeinVisualizer(320, 240, 16, 12, 1000)
  vv.setup()
  vv.streaming()

if __name__ == '__main__':
  main()
