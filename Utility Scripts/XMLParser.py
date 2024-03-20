import xml.etree.ElementTree as ET

# Objectives are to get each bounding box for each frame

# Parsing XML into Element Tree
tree = ET.parse('bww2gt.xml') # Change this line to the file name of XML


root = tree.getroot() # Each XML file needs one root

f = open("bww2gtAmended.txt", "w")

tick = 0
for frame in root.iter('frame'):

    # if tick > 5: # Change to see different stuff
    #     break

    #Fetches the frame number

    # we may want to do something like currentframenum + totalframes to match with dataset frames
    framenumber = frame.attrib['number'] + '.jpg' # may need to add zip to this.

    # Reset the lists for each frame
    heightlist = []
    widthlist = []
    xcentrelist = []
    ycentrelist = []

    #print(framenumber)

    boxExists = frame.find('objectlist/object/box')

    #print(boxExists)

    if boxExists is not None:
        # Iterate through all objects in the frame
        for box in frame[0].iter('box'):
            #print(box.attrib) # Prints the boxes for each frame
            
            # We need to convert the attributes to what TFRecords can read
            # xmin, xmax, ymin, ymax

            #print(box.attrib)

            h = float(box.attrib['h'])
            w = float(box.attrib['w'])
            xc = float(box.attrib['xc'])
            yc = float(box.attrib['yc'])

            # Converting:
            xmin = xc - (0.5*w) # Floating point but can TFRecord accept a non integer? Documentation says floating point.
            ymin = yc + (0.5*h)
            xmax = xc + (0.5*w)
            ymax = yc - (0.5*h)

            # print(f'h: {h}, w: {w}, xc: {xc}, yc: {yc}')
            # print(f'xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}')

            outputLine = framenumber + " " + str(xmin) + " " + str(xmax) + " " + str(ymin) + " " + str(ymax)
            f.write(outputLine+"\n")
            print(outputLine)

    else:
        outputLine = framenumber + " " + "nothing"
        f.write(outputLine+"\n")
        print(outputLine)

    # print("endframe\n")
    tick += 1
    
    #break
    #print(framenumber) # 0 - 1042


# print("frame:" + str(tick))

f.close()