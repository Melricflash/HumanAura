import onnx
import onnxruntime as ort
import cv2
import numpy as np
import os

'''
Configurations for the program, to be changed according to different scenarios
'''

# Path to where model is located, must point to an ONNX file (only tested with static input tensor so far)
modelPath = "ONNX Models/mobilenet2fixed.onnx"

# Path to where the test frame is stored, used for debugging, inference and building up of the program
imagePath = "testframe.jpg"

# Video path that points to the video we want to use for inference
videoPath = "Videos/Browse4.mpg" # browse4.mpg was our test set on evaluation

# Detection Threshold, affects the sensitivity of the bounding boxes that are drawn.
# Lower threshold = Less confidence needed to display BB on screen
# Between 0.0 and 1.0
# Set appropriately to avoid false detections
globalThreshold = 0.65

# Image Height and Width, currently hardcoded but to be calculated later
imHeight = 288
imWidth = 384

# Number used to determine if the euclidean distance calculated between two boxes is small enough to be determined the same person.
# Larger = larger euclidean distance to be identified as the same person.
# Needs to be set appropriately to avoid false detections
trackerThreshold = 5

# Currently unused...
currentBoundingBoxes = [] # Bounding box list for current frame
previousBoundingBoxes = [] # Bounding box list for previous frame

'''
Helper functions and used by HumanAura
'''

# Checks if the ONNX model supplied is legal and can be used
def is_onnx_model_valid(path):
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        return False
    return True

# Takes a frame and preprocesses it so it can be used to perform inference with the model
def preprocessFrame(frame):
    #print(frame.shape)

    # We expand the image to match the input shape of the model from 3 dimensions to 4.
    expanded = np.expand_dims(frame, axis=0)

    # Input to the model requires the input type to be tensor(uint8), note this is kind of misleading as it is already tensor(uint8) and run() requires a np array
    #print(expanded.dtype) # uint8

    return expanded

# Decodes the normalised bounding boxes so they are converted back into unnormalised coordinates to be able to be plotted
def boxDecoder(boxes):
    tempList = []

    tempList.append(boxes[0]*imHeight) # ymin * height
    tempList.append(boxes[1]*imWidth) # xmin * width
    tempList.append(boxes[2]*imHeight) # ymax * height
    tempList.append(boxes[3]*imWidth) # xmax * width

    #print(tempList)
    return tempList


# After getting the results from inference on frame, fetch the bounding box information and confidence scores
# Filter the relevant bounding boxes out according to the specified threshold score eg. >0.65
# Decode the bounding boxes back into coordinates that can be used later
def extractBoundingBoxes(results, threshold):
    bbinfo = results[1] # (1,100,4) # Holds all bounding boxes created by model
    confidenceinfo = results[4] #(1,) # Holds all confidences created by the model for each 100 bounding boxes

    # Filter out the relevant bounding boxes according to the threshold set in confidenceinfo and fetch the indices
    boxIndices = np.where(confidenceinfo >= threshold)[1]
    # Get the corresponding boxes from bbinfo, these will be the bounding boxes to display to opencv
    frameBoxes = bbinfo[0, boxIndices]

    # Next step is to decode the bounding boxes by multiplying the relevant widths and heights
    # Remember that order of numbers are [ymin, xmin, ymax, xmax]
    # Decode by doing x * width, y * height

    #print(len(frameBoxes))
    boundingBoxes = []
    
    if len(frameBoxes) != 0: # We dont want to perform operations if we dont have any bounding boxes to decode
        for box in frameBoxes:
            boundingBoxes.append(boxDecoder(box))

    #print(len(boundingBoxes[0])) # (2,4) Shape preserved

    return(boundingBoxes)

# Draw Bounding Boxes to the screen
def drawBoundingBoxes(boxList):
    #print(boxList)

    for box in boxList:
        # Fetch the coordinates we need for each list we have in the boxList
        ymin = int(box[0])
        xmin = int(box[1])
        ymax = int(box[2])
        xmax = int(box[3])

        # OpenCV wants the order two coordinates to draw a rectangle, (top left) and (bottom right)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Show the window with the bounding boxes applied to the image
    cv2.imshow("HumanAura", image)

    cv2.waitKey(30)  # If set to 0, need to press a key to quit

'''
Functions used for tracking boxes between frames
'''

# Get the midpoint of a box in the form of (xcentre, ycentre) from ymin, xmin, ymax, xmax
# Give just one box to this function and it will return the midpoint coordinates in a list
def getBoxMidpoint(box):
    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]

    xcentre = int(((xmin+xmax)/2))
    ycentre = int(((ymin+ymax)/2))

    midpoint = [xcentre, ycentre]

    return midpoint

# Similar to previous function, but does it for a set of bounding boxes instead of just one. (May not be used)
def getBoxListMidpoint(boxList):
    midpointList = []

    for box in boxList:
        boxMP = getBoxMidpoint(box)
        midpointList.append(boxMP)

    return midpointList

# Function to calculate the euclidean distance between two midpoints of two bounding boxes
def calculateEuclideanDistance(currentMP, previousMP):
    currentX = currentMP[0]
    currentY = currentMP[1]
    previousX = previousMP[0]
    previousY = previousMP[1]

    #print(f"{currentX}, {currentY}, {previousX}, {previousY}")
    ed = np.sqrt((currentX-previousX)**2 + (currentY-previousY)**2) # Euclidean Distance formula
    #print(ed)

    return ed



'''
Main Loop: Used for video inference
'''

# Attempt to load the specified model into memory
# Using Static Input Tensor shows warnings when loading model.
# Currently only tested with a fixed input tensor shape
sess = ort.InferenceSession(modelPath)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Load the video into memory
cap = cv2.VideoCapture(videoPath)

# Returns an error if the video has an error
if not cap.isOpened():
        print("Error opening video file.")


# Main loop to load video and perform inference using the functions
while True:
    ret, image = cap.read()

    if not ret:
        break

    newInput = preprocessFrame(image)
    result = sess.run(None, {input_name: newInput})
    boundingBoxestoDraw = extractBoundingBoxes(result, globalThreshold)
    drawBoundingBoxes(boundingBoxestoDraw)

# Delete all windows
cap.release()
cv2.destroyAllWindows()





'''
 Test Main Loop: Used for a test functions with a single frame, commented out as we work with a video not a single frame in practice
'''

# # Attempt to load the specified model into memory
# # Using Static Input Tensor shows warnings when loading model?
# sess = ort.InferenceSession(modelPath)
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name

# # Fetch the image to be passed into model
# image = cv2.imread(imagePath)

# # Send frame for pre processing
# newInput = preprocessFrame(image)

# # Performing inference using the frame
# result = sess.run(None, {input_name: newInput})

# # Get the list of bounding boxes to draw onto the image
# boundingBoxestoDraw = extractBoundingBoxes(result, globalThreshold)

# # md = getBoxMidpoint(boundingBoxestoDraw[0])
# # print(boundingBoxestoDraw[0])
# # print(md)

# mdList = getBoxListMidpoint(boundingBoxestoDraw)
# # print(mdList)
# ed = calculateEuclideanDistance(mdList[0], mdList[1])

# # Attempting to draw bounding boxes onto image
# drawBoundingBoxes(boundingBoxestoDraw)

