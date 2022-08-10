import numpy as np
import SimpleITK as sitk
from skimage import color, io
import matplotlib.pyplot as plt

def crop_red(name):
    #get red line as really bright white line
    original = io.imread(name)
    original_gray = color.rgb2gray(original)
    image = color.rgb2hsv(original)
    image = color.rgb2gray(image)
    
    #initialize and create mask
    mask = np.zeros((len(image), len(image[1])))

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] >= 0.5:
                mask[i][j] = 1
    
    for i in range(len(mask[1])):
        mask[0][i] = 0
        mask[1][i] = 0
        mask[2][i] = 0
        mask[-1][i] = 0
        mask[-2][i] = 0
        mask[-3][i] = 0
            
    for i in range(len(mask)):
        mask[i][0] = 0
        mask[i][1] = 0
        mask[i][2] = 0        
        mask[i][-1] = 0
        mask[i][-2] = 0
        mask[i][-3] = 0    
    
    for i in range(len(mask)):
        for j in range(len(mask[1]) - 1):
              if (mask[i][j] == 0) and (mask[i][j+1] ==1):
                  mask[i][j] = 1
    
    for j in range(len(mask[1])):
        start = None
        prevstart = None
        for i in range(len(mask)):
            if mask[i][j] >= 1:
                start = i
            if start != None:
               break
        if (start != None) and (start <= 100):
            if prevstart != None:
                start = prevstart
            else:
                start = 190
        
        for i in range(len(mask)):
            if (start != None) and (i >= start):
                mask[i][j] = 1
        prevstart = start

    return mask

#now need to make a mask that crops out everything below the white
def crop_white(name):
    #get red line as really bright white line
    original = io.imread(name)
    image = color.rgb2gray(original)
    
    #initialize and create mask
    mask = np.zeros((len(image), len(image[1])))
    
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] >= 0.5:
                mask[i][j] = 1
    
    for i in range(len(mask[1])):
        mask[0][i] = 0
        mask[1][i] = 0
        mask[2][i] = 0
        mask[-1][i] = 0
        mask[-2][i] = 0
        mask[-3][i] = 0
            
    for i in range(len(mask)):
        mask[i][0] = 0
        mask[i][1] = 0
        mask[i][2] = 0        
        mask[i][-1] = 0
        mask[i][-2] = 0
        mask[i][-3] = 0    
    
    for i in range(len(mask)):
        for j in range(len(mask[1]) - 1):
            if (mask[i][j] == 0) and (mask[i][j+1] ==1):
                mask[i][j] = 1
            if i >= 450:
                mask[i][j] = 0

    prevend = np.array([None, None, None])
    for j in range(len(mask[1])):
        end = None

        for i in range(len(mask)-1, -1, -1):
            if mask[i][j] >= 1:
                end = i
                if j == 4:
                    keepend = i
            if end != None:
                break
        if (end != None) and (end <= 100):
            if prevend[0] != None:
                end = prevend[0]
            else:
                end = 250
                
        if (end != None) and (j >= 3):
            if (prevend[0] == 0) and (prevend[1] == 0) and (prevend[2] == 0):
                end = keepend
        
        for i in range(len(mask)-1, -1, -1):
            if (end != None) and (i <= end):
                mask[i][j] = 1
        
        if j == 0:
            prevend[0] = end
        if j == 1:
            prevend[1] = prevend[0]
            prevend[0] = end
        if j >= 2:
            prevend[2] = prevend[1]
            prevend[1] = prevend[0]
            prevend[0] = end      
    
    for i in range(len(mask)):
        count = 0
        for j in range(len(mask[1])):         
            if mask[i][j] == 0:
                count += 1
            else:
                count = count - 1
                
            if count == 5:
                mask[i][j] = 0
            
    return mask

def crop(name):
    white = crop_white(name)
    red = crop_red(name)
    mask = np.multiply(white, red)
    img = io.imread(name)
    img = color.rgb2gray(img)
    img = np.multiply(img, mask)
    return img

