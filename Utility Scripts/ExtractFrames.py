import os
import cv2

# Declare the path of the video that we want to extract the frames from.
video = 'Browse_WhileWaiting1.mpg'

# Check if the file provided exists and can be found
print("file exists?", os.path.exists(video))
cap = cv2.VideoCapture(video) # Open the video using OpenCV

count = 0

# Sets success to true if a valid frame was read and stores the image of the frame into a variable
success, image = cap.read()

# Changes to the directory named frames if it exists, if running this, you should create a frame folder in the same location as a video to avoid clutter
os.chdir('Frames') 
print(os.getcwd())

# While there are still frames to process, write the frame to disk with the name set by the current count of the frame counter
# Also save as jpg file as this is the most compatible for images.
while success:
    cv2.imwrite("%d.jpg" % count, image)
    success, image = cap.read()
    count += 1

print("Frames Created: " + str(count)) # Tells us how many frames we wrote to disk and thus how many we end up extracting.


# count = 0
# vidcount = 0

# os.chdir('Videos')

# while vidcount < 9:
#     print("\nCount: " + str(vidcount))
#     os.chdir('../Videos2')
#     print(os.getcwd())

#     #print("file exists?", os.path.exists(str(vidcount) + '.mpg'))

#     # Switching to new video file and attempting to extract its frames

#     cap = cv2.VideoCapture(str(vidcount) + '.mpg')
#     print("file exists?", os.path.exists(str(vidcount) + '.mpg'))

#     success, image = cap.read()

#     # remember to switch directories to put the frames in the right place

#     os.chdir('../Frames')
#     print(os.getcwd())

#     while success:
#         cv2.imwrite("%d.jpg" % count, image)
#         # Read next frame
#         success, image = cap.read()
#         count += 1

#     vidcount += 1


# print("Reached End!")
# print("\nFrames created: " + str(count))

# Creates 9254 Frames