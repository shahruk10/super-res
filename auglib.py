import cv2

def rotate(x,rot,cols,rows):
    rotMat = cv2.getRotationMatrix2D((cols/2, rows/2), rot, 1)
    xRot = cv2.warpAffine(x,rotMat, (cols,rows))
    return xRot

def horFlip(x):
    return cv2.flip(x,0)

def verFlip(x):
    return cv2.flip(x,1)

