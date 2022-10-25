#####
#DETAILS
####
#Volume calculation project
#Code requires that the images are named correctly and identically exept for sequence information, which MUST BE ACCURATE


#####
#IMPORTS
#####
import numpy as np
import cv2 
import os
import sys
import random
import matplotlib.pyplot as plt
import shapely.geometry as geom
import csv
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import font
from skimage import color, io, img_as_float
from skimage.segmentation import chan_vese
from skimage.io import imread
from skimage.color import rgb2gray
from pathlib import Path

#####
#INITIAL PARAMETERS
#####
#need to get slice width, pixel size, and path to pictures
root = tk.Tk()
root.withdraw()
ft = font.nametofont("TkDefaultFont")
ft.config(size = 15)
ans = simpledialog.askstring(title="Bubble Volume Calculation", prompt='Please enter the spacing between slices, in mm, to 3 decimal places. \nThe default value is 6/128 (6mm depth, 128 scans). \n\nTo select the default value, enter 0')

if ans == '0':
    depth = 6/128
else:
    depth = ans
    
root = tk.Tk()
root.withdraw()
ans = simpledialog.askstring(title="Bubble Volume Calculation", prompt='Please enter the pixel width, in mm, to 3 decimal places. \nThe default value is 6/844 (6mm wide, 844 pixels). \n\nTo select the default value, enter 0')

if ans == '0':
    pix_width = 6/844
else:
    pix_width = ans
    
root = tk.Tk()
root.withdraw()
path = simpledialog.askstring(title="Bubble Volume Calculation", prompt='PLEASE ENSURE THE FOLLOWING FOR YOUR IMAGES:\n1) Your images for ONE eye and ONLY one eye are in one folder, with nothing else in it, except potentially this .py file\n2) Your images for ONE eye and ONLY one eye are named sequentially and otherwise identically; increasing or decreasing depth does not matter.\n3) Your images are all oriented the same way.\n\nIf your images are NOT in the same folder as this .py file, please provide the path, from home, to the folder containing your images. If your images are in this folder, enter 0\n\nOnce you have done so, you may begin by selecting your bubble of choice, or pressing SPACE If no bubble can be seen.')

if path == '0':
    path = "/home/jui434/Documents/Muni Lab/Volume/data2"
    #path = os.path.dirname(os.path.abspath(__file__))

#####
#LOADING AND SELECTING THE BUBBLE
#####

#first, load and display the FIRST image
arr = []
for subdir, dirs, files in os.walk(path): #from shaan's code; get path for every scan
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".tif") and not "llips" in filepath: #get only the scans (filetype .tif)
            arr.append(filepath)


#sort pictures in ascending order if they are not properly ordered
def myFunc(e):
  return len(e)

# use .sort() on the list and sort by the len function we created above
arr.sort()
arr.sort(key=myFunc)

#define click event
bubble_loc = [] #store location of first bubble

def bubble_select(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        bubble_loc.append((x,y)) #store coordinates

start = 999999999

#now display and register button
for item in arr: #iterate through files in array until bubble is selected
    scan = cv2.imread(item) #get cv2 scan info for picture
    cv2.imshow("scan", scan) #display picture
    cv2.setMouseCallback('scan', bubble_select) #apply click event
    cv2.waitKey()
    if bubble_loc != []: #break out of iteration once bubble is selected
        start = arr.index(item) #save starting index for bubble calculation
        break

#if no image selected, leave
if start == 999999999:
    print("No image selected. Exiting.")
    sys.exit()

#####
#CALCULATING CONTOURS
#####
areas = [] #empty array will hold areas
bubbles = [] #empty array will hold contours
bubbles_shapely = [] #empty array will hold shapely contours

#make CSV
header = ['name', 'pixels', 'area']
with open(path+'/areas.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(header)

a = 0

#we will be deconvoluting or otherwise cleaning the images as we go to improve contour selection
for i in range(start, len(arr)): #iterate through all scans from SELECTED To end
    #from shaan's code for pre-contour recognition thresholding
    scan = cv2.imread(arr[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = cv2.fastNlMeansDenoising(img,img,20,5,21)
    img = cv2.bitwise_not(img)
    blur = cv2.GaussianBlur(img,(5,5),0)
    blur = cv2.add(blur, blur)
    #cv2.imshow('blur', blur)
    #cv2.waitKey()
    if a <= 30:
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    else:
        scikittest = img_as_float(blur)
        cv = chan_vese(scikittest, mu=0.07, lambda1=1, lambda2=1.00, tol=1e-5,max_num_iter=50, dt=0.8, init_level_set="checkerboard",extended_output=True)
        cv = cv[0].astype(np.uint8)*255
        #cv2.imshow('cv1', cv)
        #cv2.waitKey()
        contours, hierarchy = cv2.findContours(cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    

    #ask if they want to review results or just get the volume
    if i == start:
        root = tk.Tk()
        root.withdraw()
        x = simpledialog.askstring(title="Bubble Volume Calculation", prompt='Do you want to walk through all the scans? y/n')
        
        if x == 'y':
            show = True
        elif x == 'n':
            show = False  
    
        #select bubble contour
        #for FIRST image, this is based on click (bubble_loc)
        coord = bubble_loc[0]
        print(coord)
        
        #find the contour of SMALLEST AREA that contains the coord
        asmallest = 999999999 #placeholder
        contour_bubble = None #placeholder
        contour_shapely = None #placeholder
        for cnt in contours:
            cnt = cnt
            if len(cnt) >= 3: #lines cannot become polygons
                a = cv2.contourArea(cnt)
                #convert to shapely object
                shapely_contour = np.squeeze(cnt)
                polygon = geom.Polygon(shapely_contour)
                #convert bubble loc to shapely point
                loc = geom.Point(coord)
                                
                #see if contour is close enough to click
                #contains fails due to I Hate Computers
                if polygon.exterior.distance(loc) < 5:
                    #see if area is smallest
                    if a < asmallest:
                        #conditions met? take contour
                        asmallest = a
                        contour_bubble = cnt
                        contour_shapely = polygon
                        M = cv2.moments(cnt)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        coord = (cx, cy)

        #store chosen contour
        areas.append(asmallest)
        a = asmallest
        bubbles.append(contour_bubble)  
        bubbles_shapely.append(contour_shapely)
        with open(path+'/areas.csv', 'a') as f: #save results
            writer = csv.writer(f)
            line = [arr[i], asmallest, asmallest*pix_width*pix_width]
            writer.writerow(line)
        
    else:
        #for remaining images, this is based on previous coordinates
        asmallest = 999999999 #placeholder
        contour_bubble = None #placeholder
        contour_shapely = None #placeholder
        prev = bubbles_shapely[-1]
        
        #we want the search to end if there is no bubble
        bubble_here = False
        print(coord)
        
        #this time, we check for overlap with previous contour, and containing point, and smallest 		area
        for cnt in contours:
            cnt = cnt
            if len(cnt) >= 3: #lines cannot become polygons
                a = cv2.contourArea(cnt)
                #convert to shapely object
                shapely_contour = np.squeeze(cnt)
                polygon = geom.Polygon(shapely_contour)
                #convert bubble loc to shapely point
                loc = geom.Point(coord)
                                
                #see if contour contains click
                if loc.within(polygon):
                    #see if overlap with previous contour
                    if prev.intersects(polygon) != None:
                        #see if area is smallest
                        if a < asmallest:
                            asmallest = a
                            contour_bubble = cnt
                            contour_shapely = polygon
                            M = cv2.moments(cnt)
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                            coord = (cx, cy)                            
                            bubble_here = True
        
        #break out of the loop if the chosen contour is stupidly large ie: not a bubble                    
        if asmallest >= 10*max(areas):
            bubble_here = False
               
        if bubble_here:   
            a = asmallest        
            areas.append(asmallest)  
            bubbles.append(contour_bubble)
            bubbles_shapely.append(contour_shapely)
            with open(path+'/areas.csv', 'a') as f: #save results
                writer = csv.writer(f)
                line = [arr[i], asmallest, asmallest*pix_width*pix_width]       
                writer.writerow(line)     
        else:
            break
    
    if show: #if they WANT to review, show them each scan for vetting
        first, second = contour_shapely.exterior.xy
        plt.plot(first, second, zorder = 1000000, c = 'g', linewidth = 1)
        plt.imshow(scan, zorder = 1)
        name = arr[i][:-4]+"_selected.png"
        plt.axis("off")
        plt.savefig(name)
        plt.show()
    else: #regardless, draw and save the image
        plt.clf()
        first, second = contour_shapely.exterior.xy
        plt.plot(first, second, zorder = 1000000, c = 'g', linewidth = 1)
        plt.imshow(scan, zorder = 1)
        plt.axis("off")        
        name = arr[i][:-4]+"_selected.png"
        plt.savefig(name)

#####
#CALCULATE AREA
#####
#use conversion factors
scaled_area = []
for item in areas:
    scaled_area.append(item * pix_width**2)

#volume has n slices and n depths
volume = 0 #mm
for scaled in scaled_area:
    volume += scaled * depth
volume = str(volume)

with open(path+'/areas.csv', 'a') as f: #save results
    writer = csv.writer(f)
    line = ['final volume',float(volume)]       
    writer.writerow(line)    

root = tk.Tk() 
root.withdraw()
font1 = font.Font(name='TkCaptionFont', exists=True)
font1.config(family='Calibri', size=15)
messagebox.showinfo(message="The bubble volume is "+volume+"mm^3")
