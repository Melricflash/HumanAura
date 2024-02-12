import onnx
import onnxruntime as ort
import cv2
import numpy as np
import os

'''
Configurations for the program, to be changed according to different scenarios
'''

print(os.getcwd())

# Path to where model is located, must point to an ONNX file (only tested with static input tensor so far)
modelPath = "../ONNX Models/mobilenet2fixed.onnx"

# Path to where the test frame is stored, used for debugging, inference and building up of the program
imagePath = "../testframe.jpg"

# Video path that points to the video we want to use for inference
videoPath = "../Videos/Walk1.mpg" # browse4.mpg was our test set on evaluation

# Detection Threshold, affects the sensitivity of the bounding boxes that are drawn.
# Lower threshold = Less confidence needed to display BB on screen
# Between 0.0 and 1.0
# Set appropriately to avoid false detections

# Originally set to 0.65, MobileNet may do better with a lower detection threshold
globalThreshold = 0.4

# Image Height and Width, currently hardcoded but to be calculated later
imHeight = 288
imWidth = 384

# Number used to determine if the euclidean distance calculated between two boxes is small enough to be determined the same person.
# Larger = larger euclidean distance to be identified as the same person.
# Needs to be set appropriately to avoid false detections

# Originally set to 30, may do better with a larger number regarding mobilenet
trackerThreshold = 100 # 100 Seems to do okay, still suffers from flickering

# Currently unused...
currentBoundingBoxes = [] # Bounding box list for current frame
previousBoundingBoxes = [] # Bounding box list for previous frame

loggedBoxes = {}

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

# Draw Bounding Boxes to the screen, currently being replaced with one that incorporates tracking as well
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
        # Draws text on the bounding boxes, replace test with the time counter when function implemented
        cv2.putText(image, 'test', (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Show the window with the bounding boxes applied to the image
    cv2.imshow("HumanAura", image)

    cv2.waitKey(0)  # If set to 0, need to press a key to quit, good for debugging

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

    xcentre = ((xmin+xmax)/2)
    ycentre = ((ymin+ymax)/2)

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

def trackAndDraw(boxList):

    for box in boxList:
        #print(box)

        # Fetch the coordinates we need for each list we have in the boxList
        ymin = int(box[0])
        xmin = int(box[1])
        ymax = int(box[2])
        xmax = int(box[3])

        # OpenCV wants the order two coordinates to draw a rectangle, (top left) and (bottom right)
        # We can draw the rectangle now and do the counter later in the loop
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    
       
        currentMP = getBoxMidpoint(box)

        # Only want to compare midpoints if the dictionary is populated
        if len(loggedBoxes) > 0:
            #print(loggedBoxes)

            matchFound = False
            # Compare the current current box midpoint to all tracked bounding boxes in the dictionary, break if we find a match and update the item in the dictionary
            for previousMP, MPValues in loggedBoxes.items():
                # Retrieve info out of the value tuple, one is used for tracking time, second is an expiry to remove old values when needed
                trackCounter, frameCounter = MPValues
                
                # Calculate the euclidean distance between the currentMP and selected entry in the dictionary
                eDistance = calculateEuclideanDistance(currentMP, previousMP)
                #print(eDistance)

                if eDistance < trackerThreshold:
                    print("Match was found for box!")
                    matchFound = True
                    # Increment the trackCounter
                    loggedBoxes[previousMP] = (trackCounter + 1, FCounter)
                    cv2.putText(image, str(trackCounter), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                    break
            
            # FrameCounter will only update if a match was found for that box, allows us to figure out if we need to prune
            if not matchFound:
                print("No match was found for box!")
                # No match for the frame was found, we add it to the database to track later
                #loggedBoxes[tuple(currentMP)] = (0, newFrameCounter + 1)
                loggedBoxes[tuple(currentMP)] = (0, FCounter) # Set 0 to set time tracked as 0s, recorded frame as last recorded frame

        else:
            # Store into dictionary as our dictionary is currently empty
            print("loggedBoxes currently empty, populating...")
            boxMidpoint = tuple(getBoxMidpoint(box))
            #print(boxMidpoint)
            loggedBoxes[boxMidpoint] = (0,FCounter) # Store as a tuple, when we want to modify it, we will update the whole thing.


    # Length of dictionary increases over time, we need to prune old boxes accordingly.
    prunableEntries = [] # List to hold the entries we need to delete from the dictionary to stop it from growing

    # Go through each item in the dictionary and check if it should be pruned
    for midpoint, mpvalues in loggedBoxes.items():
        trackC, recordedFrame = mpvalues

        #print(f'Recorded: {recordedFrame}') # Debug
        #print(f'FC: {FCounter}') # Debug

        # If the last recorded frame is less than the current frame we should get rid of it as it is irrelevant for tracking (adjustable?)
        if recordedFrame < FCounter - 1:

            #print(f'Appending, {recordedFrame} < {FCounter-1}') # Debug

            prunableEntries.append(midpoint) # Add the midpoint to delete into the array

    for entry in prunableEntries:
        del loggedBoxes[entry] # Go through the items we need to prune from the dictionary

    #print(len(loggedBoxes))
    #print(loggedBoxes)

    cv2.imshow("HumanAura", image)
    cv2.waitKey(30)
    



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

FCounter = 0

# Main loop to load video and perform inference using the functions
while True:
    ret, image = cap.read()

    if not ret:
        break

    newInput = preprocessFrame(image)
    result = sess.run(None, {input_name: newInput})
    boundingBoxestoDraw = extractBoundingBoxes(result, globalThreshold)

    #drawBoundingBoxes(boundingBoxestoDraw)
    trackAndDraw(boundingBoxestoDraw)

    FCounter += 1

    # Change to debug more frames
    # if FCounter >= 5:
    #     break

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

