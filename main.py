##Imports for robot functionality
from jetbotmini import Camera
from jetbotmini import bgr8_to_jpeg
from jetbotmini import Robot
import time
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import cv2
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display

##Setup camera
camera = Camera.instance(width=300, height=300)

##Global parameters for FSM
global color_lower
global color_upper
global searching_target 
global approaching_target 
global cycle_done 
searching_target = 1
approaching_target = 0
cycle_done = 0

##Setup colour bound arrays. This can be modified/expanded/randomized for more fun
# Define 5 different triplets and their corresponding color names
lower_color_bounds = np.array([
    #blue
     [100, 100, 100],
    #green
     [35, 43, 46],
    #yellow
     [21, 80, 80],
    #orange
     [13,100,100],
    #red
     [0,90,90],
    #purple
    [129,70,70]
])
upper_color_bounds = np.array([
    #blue
    [128, 255, 255],
    #green
    [80, 255, 255],
    #yellow
    [35, 255, 255],
    #orange
    [20, 255, 255],
    #red
    [7,255,255],
    #purple
    [170,255,255]
])

##Initialize colours to initial state
color_lower = lower_color_bounds[0]
color_upper = upper_color_bounds[0]
color_index = 0

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

##Define robot object and its activity functions
robot = Robot()

# Function to rotate the robot slowly
def rotate_robot():
    robot.left(0.3)
    
def rotate_robot_fast():
    robot.right(0.6)
    
# Function to stop the robot
def stop_robot():
    robot.stop()


image_widget = widgets.Image(format='jpeg', width=300, height=300)
speed_widget = widgets.FloatSlider(value=0.4, min=0.0, max=1.0, description='speed')
turn_gain_widget = widgets.FloatSlider(value=0.5, min=0.0, max=2.0, description='turn gain')
center_x = 0
display(widgets.VBox([
    widgets.HBox([image_widget]),
    speed_widget,
    turn_gain_widget
]))
width = int(image_widget.width)
height = int(image_widget.height)      
def execute(change):
    global color_lower, color_upper, color_index, approaching_target, searching_target, cycle_done
    
    color_upper_np = np.array(color_upper)
    color_lower_np = np.array(color_lower)
    # Get new frame from the camera, fit it to size 300*300 pixels and add Gaussian noise 
    frame = camera.value
    frame = cv2.resize(frame, (300, 300))
    frame =cv2.GaussianBlur(frame,(5,5),0)  
    # Get hsv values of the frame and filter out any parts of the frame not within the range of the current searched colour. 
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,color_lower_np,color_upper_np)  
    # Use opening operator(erode+dilate) to smooth contours and separate objects from each other
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    mask=cv2.GaussianBlur(mask,(3,3),0)   
    # Detect all objects which fit the current colour in the frame
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] 
    # Control jetbotmini to follow the set object
    if len(cnts)>0:
        cnt = max (cnts,key=cv2.contourArea) 
        (color_x,color_y),color_radius=cv2.minEnclosingCircle(cnt)
        # If the largest object of our colour is within our size bounds, start approaching it
        if (color_radius > 10) & (color_radius < 100):
            #Set state to Approaching Target
            approaching_target = 1
            searching_target = 0
            # Mark the detected color
            cv2.circle(frame,(int(color_x),int(color_y)),int(color_radius),(255,0,255),2)  
            # Correct location according to the error between the middle of the frame and the middle of the object
            center_x = (150 - color_x)/150
            robot.set_motors(
                float(speed_widget.value - turn_gain_widget.value * center_x),
                float(speed_widget.value + turn_gain_widget.value * center_x)
            )    
        # If the object was reached, move to the next colour
        elif (color_radius >= 100):
            #If target was just reached, stop and change state to start searching
            if (approaching_target):
                stop_robot()
                approaching_target = 0
                searching_target = 1  
                if (color_index == len(lower_color_bounds) - 1):
                    if not cycle_done:
                        rotate_robot_fast()
                        time.sleep(5)
                        cycle_done = 1
                    stop_robot()
                    time.sleep(10)
                color_index = (color_index + 1)%len(lower_color_bounds)
                color_lower= lower_color_bounds[color_index]
                color_upper = upper_color_bounds[color_index]
                cycle_done = 0
            rotate_robot()
        else:
            rotate_robot()
    # If no target is detected, rotate
    else:
        rotate_robot()
     
    # Update image display to widget
    image_widget.value = bgr8_to_jpeg(frame)  
execute({'new': camera.value})
#if (searching_target):
 #   stop_robot()
  #  time.sleep(5)
   # rotate_robot()
    #time.sleep(5)
    
