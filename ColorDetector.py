# coding=utf-8
import cv2
import numpy as np
import os

lookForColor = np.array((0, 0, 0), dtype=np.uint8) # The color to look for as a numpy(BGR) array

# TODO: skulle även kunna ha en där jag har att man varierar ett visst antal enheter, vilka
# distrubieras proportionerligt beroende på hur stor andel som är i B/G/R.
# aVUnits = 25 # The units that are to be distrubuted among B/G/R based on previous distrubution
aVUnits = 50 # The units that are to be distrubuted among B/G/R based on previous distrubution

# allowedVariancePercent = 0.025 # The shade of the color are allowed to vary x percent
allowedVariancePercent = 0.10 # The shade of the color are allowed to vary x percent
allowedVariance = np.array((10, 10, 10), dtype=np.uint8) # Allowed variance for blue, green, and red values

lookForColorRadius = 3 # How many pixels on each side of the center pixel we will take together to get
# average value which becomes the lookForColor value

# Color detector, click on a part of the image to set the color to search for.
# Allow some variance and then mark all parts of the image that is of that color, or
# blacken all parts which is not of that color within the allowed variance.

# frame is a numpy array with dimensions X x Y x BGR
# ret is a bool indicating success/failure

# Callback function
# This function sets the color we want to detect to what ever color exists at the x, y coordinates that
# the user pressed in the video frame
def detectColor(event,x,y,flags,param):
    print "Inside Callback function" # TODO: REMOVE
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print "Setting the color[" + str(x) + "][" + str(y) + "]: " # TODO: REMOVE
        global lookForColor
        # lookForColor = frame[y][x]

        # Instead of simply taking the one pixels color we take the pixels within z cells from
        # that pixel (including the pixel itself) and then we average those values
        summedList = [0, 0, 0]
        for newY in range(y-lookForColorRadius, y+lookForColorRadius):
            for newX in range(x-lookForColorRadius, x+lookForColorRadius):
                summedList[0] += frame[newY][newX][0]
                summedList[1] += frame[newY][newX][1]
                summedList[2] += frame[newY][newX][2]

        # average values
        lookForColor = [(s / (4 * lookForColorRadius * lookForColorRadius)) for s in summedList]

        print "B: ", lookForColor[0], " G: ", lookForColor[1], " R: ", lookForColor[2] # TODO: REMOVE
        global colorSet
        colorSet = True

# The waitKey function for 64 bit processors
def waitCVx64(time):
    return cv2.waitKey(time) & 0xFF

# Finds the searched for color and marks them in the image(img)
# @img - the image/frame in which we are searching for the specific color
# @color - the color we are looking for as a numpy array of size 3 (BGR)
# @variance - the allowed color variance as a numpy array of size 3 (BGR)
def findColor(img):
    # go through the img array and for all values that does not fit the allowed values,
    # we set them to black, or maybe grayscale them?

    # if grayscale, then grayscale the entire picture and keep the copy to get
    # only the values for indices outside the allowed interval

    # The solution below darkens all non-looked for color areas
    # TODO: check the first pixel in groups instead...
    for yCoord in range(len(img)):
        for xCoord in range(0, len(img[yCoord]), 10):
            if(not(isWithinInterval(img[yCoord][xCoord]))):
                # img[yCoord][xCoord][0] = img[yCoord][xCoord][0] * 0.5
                # img[yCoord][xCoord][1] = img[yCoord][xCoord][1] * 0.5
                # img[yCoord][xCoord][2] = img[yCoord][xCoord][2] * 0.5
                img[yCoord][xCoord:xCoord+10] = img[yCoord][xCoord:xCoord+10] * 0.5
            else:
                img[yCoord][xCoord:xCoord+10] = img[yCoord][xCoord:xCoord+10] * [2, 1, 1]
                # if(img[yCoord][xCoord][0] < 230):
                #     # img[yCoord][xCoord][0] = img[yCoord][xCoord][0] + 20
                #     img[yCoord][xCoord:10][0] = 255
                # if(img[yCoord][xCoord][1] < 230):
                #     img[yCoord][xCoord][1] = img[yCoord][xCoord][1] + 20
                # if(img[yCoord][xCoord][2] < 230):
                #     img[yCoord][xCoord][2] = img[yCoord][xCoord][2] + 20


# Returns true if all value (BGR) for the imgColorArray are within the allowed interval
def isWithinInterval(imgColorArray):
    # return isWithinIntervalExact(imgColorArray)
    # return isWithinIntervalVariance(imgColorArray)
    # return isWithinIntervalPercent(imgColorArray)
    return isWithinIntervalUnits(imgColorArray)

# Returns true if all value (BGR) for the imgColorArray are within the allowed interval as
# specified by only lookForColor
def isWithinIntervalExact(imgColorArray):
    if((imgColorArray[0] == lookForColor[0])):
       if((imgColorArray[1] == lookForColor[1])):
          if((imgColorArray[2] == lookForColor[2])):
             return True

    return False

# Returns true if all value (BGR) for the imgColorArray are within the allowed interval as
# specified by lookForColor and allowedVariance
def isWithinIntervalVariance(imgColorArray):
    if((imgColorArray[0] <= lookForColor[0] + allowedVariance[0]) and
       (imgColorArray[0] >= lookForColor[0] - allowedVariance[0])):
       if((imgColorArray[1] <= lookForColor[1] + allowedVariance[1]) and
          (imgColorArray[1] >= lookForColor[1] - allowedVariance[1])):
          if((imgColorArray[2] <= lookForColor[2] + allowedVariance[2]) and
             (imgColorArray[2] >= lookForColor[2] - allowedVariance[2])):
             return True

    return False

# Returns true if all value (BGR) for the imgColorArray are within the allowed interval as
# specified by lookForColor and allowedVariance
def isWithinIntervalPercent(imgColorArray):
    # print "Blue max: " + str(lookForColor[0] * (1 + allowedVariancePercent))
    # print "Blue min: " + str(lookForColor[0] * (1 - allowedVariancePercent))
    # print "Green max: " + str(lookForColor[1] * (1 - allowedVariancePercent))
    # print "Green min: " + str(lookForColor[1] * (1 - allowedVariancePercent))
    # print "Red max: " + str(lookForColor[2] * (1 - allowedVariancePercent))
    # print "Red min: " + str(lookForColor[2] * (1 - allowedVariancePercent))
    if((imgColorArray[0] <= lookForColor[0] * (1 + allowedVariancePercent)) and
       (imgColorArray[0] >= lookForColor[0] * (1 - allowedVariancePercent))):
       if((imgColorArray[1] <= lookForColor[1] * (1 + allowedVariancePercent)) and
          (imgColorArray[1] >= lookForColor[1] * (1 - allowedVariancePercent))):
          if((imgColorArray[2] <= lookForColor[2] * (1 + allowedVariancePercent)) and
             (imgColorArray[2] >= lookForColor[2] * (1 - allowedVariancePercent))):
             return True

    return False

# Returns true if all value (BGR) for the imgColorArray are within the allowed interval as
# specified by lookForColor and allowedVariance
def isWithinIntervalUnits(imgColorArray):
    slfc = float(sum(lookForColor))
    if((imgColorArray[0] <= lookForColor[0] + (aVUnits * lookForColor[0]/slfc)) and
       (imgColorArray[0] >= lookForColor[0] - (aVUnits * lookForColor[0]/slfc))):
        if((imgColorArray[1] <= lookForColor[1] + (aVUnits * lookForColor[1]/slfc)) and
           (imgColorArray[1] >= lookForColor[1] - (aVUnits * lookForColor[1]/slfc))):
            if((imgColorArray[2] <= lookForColor[2] + (aVUnits * lookForColor[2]/slfc)) and
               (imgColorArray[2] >= lookForColor[2] - (aVUnits * lookForColor[2]/slfc))):
                return True

    return False




if __name__ == '__main__':
    # cap = cv2.VideoCapture("testVideo.mp4") # Capture video from file
    cap = cv2.VideoCapture(0) # Capture video from cam
    callbackSet = False
    colorSet = False
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If we are currently the looking for color, then look for the color and
        # edit the frame accordingly
        if(colorSet):
            findColor(frame) # TODO: MAYBE NOT DO THIS EVERY LOOP? for speed

        # Display the resulting frame
        cv2.imshow('frame',frame)

        if(not(callbackSet)):
            cv2.setMouseCallback('frame',detectColor) # create a mouse callback on the given window
            callbackSet = True

        # If q is pressed we quit
        pressedKey = waitCVx64(120)
        if pressedKey == ord('q'):
            break
        elif pressedKey == ord('w'):
            colorSet = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
