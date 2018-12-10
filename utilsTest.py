import numpy as np
import cv2

def bicubicUp(segments, scale=2):
    upSamps = []
    for s in segments:
        u = cv2.resize(s, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
        upSamps.append(u)
    return np.asarray(upSamps)

def getImageSegments(img, window, step):
    ''' divides image into segments/patches of given window window. The crop window for the segment
    is moved with the specified step - so overlapping segments possible by setting step to
    a value less than crop window window. The segments are returned as a numpy array, with segments
    along the first axis. The first entry corresponds to the segment in the top left corner. The 
    order is left to right, then top to bottom If the image doesn't neatly divide into equal segments, 
    the segments are clipped to the edges and expanded in the opposite side to maintain the window. '''

    segments = []

    for rdx in range(0, img.shape[0], step):
        for cdx in range(0, img.shape[1], step):
            if cdx+window < img.shape[1] and rdx+window < img.shape[0]:
                # window doesn't exceed image edges in any direction 
                segments.append(img[rdx:rdx+window, cdx:cdx+window])
            elif cdx+window > img.shape[1] and rdx+window < img.shape[0]:
                 # window exceeds in x direction
                segments.append(img[rdx:rdx+window, -window:img.shape[1]])
            elif cdx+window < img.shape[1] and rdx+window > img.shape[0]:
                # window exceeds in y direction
                segments.append(img[-window:img.shape[0], cdx:cdx+window])
            elif cdx+window > img.shape[1] and rdx+window > img.shape[0]:
                # window exceeds in both directions
                segments.append(img[-window:img.shape[0], -window:img.shape[1]])

    return np.array(segments)

def upscaleSegments(model, segments, batchsize=200,denoise = True):
    ''' passes segments through model(s) and 
    returns upscaled segments as floating arrays 
    with values between 0.0 and 255.0 '''
    
    if denoise:
        segments = bicubicUp(segments,scale =2)

    if np.max(segments) > 1:
        segments = np.float32(segments)
        segments /= 255.0

    if not isinstance(model, list):
        model = [model]

    outputs = None
    for m in model:
        y = m.predict(segments,batch_size=batchsize,verbose=True)

        if outputs is None:
            outputs = y
        else:
            outputs += y
    
    outputs = outputs / len(model)
    outputs *= 255.0
    
    return outputs

def enhance(model, img, scale, window, step):

    ''' takes a single image and enhances it by passing it through
    the provided model(s). '''
    
    segments = getImageSegments(img,window,step)
    scaledSegments = upscaleSegments(model, segments)

    if len(img.shape)==3:    # color image
        enhancedImg = np.zeros(( int(img.shape[0] * scale), int(img.shape[1] * scale), img.shape[2]))
    elif len(img.shape)==2:  # bw image
        enhancedImg = np.zeros(( int(img.shape[0] * scale), int(img.shape[1] * scale),))

    # array to keep track of where segments overlap
    overlapMask = np.zeros_like(enhancedImg)

    newHeight = int(img.shape[0]*scale)
    newWidth = int(img.shape[1]*scale)
    newStep = int(step * scale)
    newWindow = int(window * scale)
    sidx = 0
    for rdx in range(0, newHeight, newStep):
        for cdx in range(0, newWidth, newStep):
            if cdx+newWindow < newWidth and rdx+newWindow < newHeight:
                # newWindow doesn't exceed image edges in any direction 
                enhancedImg[rdx:rdx+newWindow, cdx:cdx+newWindow] = scaledSegments[sidx,...]
                overlapMask[rdx:rdx+newWindow, cdx:cdx+newWindow] = np.ones_like(scaledSegments[sidx,...])
                sidx += 1
            elif cdx+newWindow > newWidth and rdx+newWindow < newHeight:
                 # newWindow exceeds in x direction
                enhancedImg[rdx:rdx+newWindow, -newWindow:newWidth] = scaledSegments[sidx,...]
                overlapMask[rdx:rdx+newWindow, -newWindow:newWidth] = np.ones_like(scaledSegments[sidx,...])
                sidx += 1
            elif cdx+newWindow < newWidth and rdx+newWindow > newHeight:
                # newWindow exceeds in y direction
                enhancedImg[-newWindow:newHeight, cdx:cdx+newWindow] = scaledSegments[sidx,...]
                overlapMask[-newWindow:newHeight, cdx:cdx+newWindow] = np.ones_like(scaledSegments[sidx,...])
                sidx += 1
            elif cdx+newWindow > newWidth and rdx+newWindow > newHeight:
                # newWindow exceeds in both directions
                enhancedImg[-newWindow:newHeight, -newWindow:newWidth] = scaledSegments[sidx,...]
                overlapMask[-newWindow:newHeight, -newWindow:newWidth] = np.ones_like(scaledSegments[sidx,...])
                sidx += 1

    assert np.min(overlapMask) > 0, "ERROR:something went wrong when merging scaled segments"

    enhancedImg = np.divide(enhancedImg, overlapMask)
    enhancedImg = np.uint8(enhancedImg)

    return enhancedImg

def calcPSNR(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))
    # PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    return 20*np.log10(255)-10. * np.log10(np.mean(np.square(y_pred - y_true)))