# Python program to explain cv2.resizeWindow() method 

# Importing cv2 
import cv2 

# Path 
path = 'geeks14.png'

# Reading an image in default mode 
image = cv2.imread(path) 

# Naming a window 
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
cv2.namedWindow("Resized_Window2", cv2.WINDOW_NORMAL) 
cv2.namedWindow("Resized_Window3", cv2.WINDOW_NORMAL) 
# Using resizeWindow() 
cv2.resizeWindow("Resized_Window", 300, 700)
cv2.resizeWindow("Resized_Window2", 600, 700) 
cv2.resizeWindow("Resized_Window3", 900, 700) 

# Displaying the image 
cv2.imshow("Resized_Window", image)
cv2.imshow("Resized_Window2", image)
cv2.imshow("Resized_Window3", image)


cv2.waitKey(0) 


