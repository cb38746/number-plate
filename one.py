import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

# OpenCV-Python is a library of Python bindings designed to solve computer vision problems.
# cv2.imread() method loads an image from the specified file.
# If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format) then this method returns an empty matrix.
# All three types of flags are described below:
# cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
# cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
# cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.
img = cv2.imread('4.jpg',cv2.IMREAD_COLOR)


# cv2.resize() function is that the tuple passed for determining the size of new image ((620, 480) in this case)
# follows the order (width, height)
img = cv2.resize(img, (620,480) )


# cv2.cvtColor funtion returns the image after changing it's color space
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale

# A bilateral filter is used for smoothening images and reducing noise, while preserving edges
# bilateralFilter() uses the following arguments:
# Image Source
# d: Diameter of each pixel neighborhood.
# sigmaColor: Value of \sigma in the color space. The greater the value, the colors farther to each other will start to get mixed.
# sigmaColor: Value of \sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise

# OpenCV has in-built function cv2.Canny() which takes our input image as first argument and its aperture size(min value and max value) as last two arguments
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection


# contours means an outline representing or bounding the shape or form of something.
# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
# Finding Contours 
# Use a copy of the image e.g. edged.copy() since findContours alters the image
# edged.copy() creating a copy of image named 'edged'
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts) #where cnts is the variable in which contours are stored

#sorts contours according to their area from largest to smallest
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]


screenCnt = None

# loop over our contours
for c in cnts:
# approximate the contour
# Contour Perimeter: It is also called arc length.
# It can be found out using cv2.arcLength() function.
# Second argument specify whether shape is a closed contour (if passed True), or just a curve.
 peri = cv2.arcLength(c, True)


 
 
# Contour Approximation
# It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify.
# It is an implementation of Douglas-Peucker algorithm. Check the wikipedia page for algorithm and demonstration.

# To understand this, suppose you are trying to find a square in an image,
# but due to some problems in the image, you didn’t get a perfect square, but a “bad shape” (As shown in first image below).
# Now you can use this function to approximate the shape.
# In this, second argument is called epsilon, which is maximum distance from contour to approximated contour.
# It is an accuracy parameter. A wise selection of epsilon is needed to get the correct output.
 approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 


#   we will loop though all the results and check which has a rectangle shape contour with four sides and closed figure.
# Since a license plate would definitely be a rectangle four sided figure
 # if our approximated contour has four points, then
 # we can assume that we have found our screen
 if len(approx) == 4:
  screenCnt = approx
  break

if screenCnt is None:
 detected = 0
 print "No contour detected"
else:
 detected = 1


# To draw the contours, cv2.drawContours function is used.
# It can also be used to draw any shape provided you have its boundary points. 
# Its first argument is source image, second argument is the contours which should be passed as a Python list,
# third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
if detected == 1:
 cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

#Read the number plate
# pytesseract.image_to_string( ) is used to extract text from Cropped image
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Detected Number is:",text)




# cv2.imshow() is used to display an image in a window. The window automatically fits to the image size.
# First argument is a window name which is a string. second argument is our image.
# You can create as many windows as you wish, but with different window names.

cv2.imshow('image',img)
cv2.imshow('Cropped',Cropped)
# cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds.
# The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues.
# If 0 is passed, it waits indefinitely for a key stroke.
# cv2.waitKey(0) will display the window infinitely until any keypress (it is udes to display images)
cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
# If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument...
cv2.destroyAllWindows()
