import onnx
import onnxruntime as ort
import cv2
import numpy as np
import os
# C:\Users\Melric\Documents\Uni\Year 3\Final Year Project\HumanAura is the current working directory for some reason


# Change to use different models in inference
modelPath = "ONNX Models/mobilenet2.onnx"
modelPath2 = "ONNX Models/mobilenet2fixed.onnx"

# Image path that points to the test frame to be used
imagePath = "testframe.jpg"

# Video path that points to the video we want to use
videoPath = "Videos/Browse4.mpg" # browse4.mpg was our test set on evaluation

# Detection Threshold, affects the sensitivity of the bounding boxes that are drawn. Lower threshold = Less confidence needed to display BB
# Between 0.0 and 1.0
globalThreshold = 0.65

# Image Height and Width, currently hardcoded but to be calculated later
imHeight = 288
imWidth = 384

# Checks if the ONNX model supplied is legal and can be used
def is_onnx_model_valid(path):
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        return False
    return True

# MobileNet2 (No Fix)
# Input Name: input_tensor
# Input Shape: [1, 'unk__1092', 'unk__1093', 3]
# Input Type: tensor(uint8)

def preprocessFrame(frame):
    #print(frame.shape)

    # We expand the image to match the input shape of the model from 3 dimensions to 4.
    expanded = np.expand_dims(frame, axis=0)

    # Input to the model requires the input type to be tensor(uint8), note this is kind of misleading as it is already tensor(uint8) and run() requires a np array
    #print(expanded.dtype) # uint8

    return expanded

# Decodes the bounding boxes so they are converted back into coordinates
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

    cv2.waitKey(10)  # If set to 0, need to press a key to quit



# Main Loop Here


# Attempt to load the specified model into memory
# Using Static Input Tensor shows warnings when loading model?
sess = ort.InferenceSession(modelPath2)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

''' 
Original Test code to see if I can draw bounding boxes onto a single frame, obsolete as I can do this in a video now
'''

# # Fetch the image to be passed into model
# image = cv2.imread(imagePath)

# newInput = preprocessFrame(image)

# # Output Name: detection_anchor_indices
# # Output Shape: [1, 'unk__1094']
# # Output Type: tensor(float)

# # Performing inference using the frame
# result = sess.run(None, {input_name: newInput})

# boundingBoxestoDraw = extractBoundingBoxes(result, globalThreshold)

# #print(boundingBoxestoDraw)

# # Attempting to draw bounding boxes onto image
# drawBoundingBoxes(boundingBoxestoDraw)

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




