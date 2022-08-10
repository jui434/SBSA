#####
#DETAILS
####
#Volume calculation project
#Code requires that the images are named correctly and identically exept for sequence information, which MUST BE ACCURATE
#Code by Dr. Shaan Bhambra (image cleanup and contour calculation) and Julie Midroni (bubble contour selection, forward propagation and user interface)
#in other words, please don't ask julie anything about image cleanup/deconvolution because she has absolutely no idea how shaan made it work. props to him.

#####
#IMPORTS
#####
import numpy as np
import masking as mask
import cv2 
import os
import sys
import matplotlib.pyplot as plt
import shapely.geometry as geom
import csv
from skimage import color, io
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import font
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
rootdir = simpledialog.askstring(title="Bubble Volume Calculation", prompt='PLEASE ENSURE THE FOLLOWING FOR YOUR IMAGES:\n1) Your images for ONE eye and ONLY one eye are in one folder, with nothing else in it, except potentially this .py file\n2) Your images for ONE eye and ONLY one eye are named sequentially and otherwise identically; increasing or decreasing depth does not matter.\n3) Your images are all oriented the same way.\n\nIf your images are NOT in the same folder as this .py file, please provide the path, from home, to the folder containing your images. If your images are in this folder, enter 0\n\nOnce you have done so, you may begin by selecting your bubble of choice, or pressing SPACE If no bubble can be seen.')

if rootdir == '0':
    rootdir = Path("./")

#####
#IMPORT BOX
#####
'''
box = []
file_list = [f for f in os.listdir(rootdir)]
filenames = []
for thing in file_list:
    if '.jpg' in thing:
        truename = str(thing)
        filenames.append(int(truename[:-4]))

filenames.sort()

for name in filenames:
     picture = io.imread(str(name) + '.jpg')
     picture = color.rgb2gray(picture)
     box.append(picture)
     print(name)

box = np.array(box)
np.save('box', box)'''

box = np.load('box.npy')

#####
#SLICE THE BOX HORIZONTALLY INSTEAD
#####
reslice = []

for k in range(0, len(box[1])):
    newsheet = box[:,k,:]
    reslice.append(newsheet)

arr = np.array(reslice)

#slice the box and see how it looks, tapping through all
#adapt code below to check images at


#####
#LOADING AND SELECTING THE BUBBLE
#####



#define click event
bubble_loc = [] #store location of first bubble

def bubble_select(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        bubble_loc.append((x,y)) #store coordinates

#make CSV
header = ['name', 'pixels', 'area']
with open('./areas.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(header)

start = 999999999
'''
for i in range(len(arr)):
    tosave = arr[i,:,:]
    plt.imshow(tosave)
    plt.axis('off')
    plt.savefig('cv2op/' + str(i)+'cv2'+'.jpg', bbox_inches='tight', pad_inches = 0)
'''

def iterate(start_pts):
    endv = 0
    for item in start_pts:
        beginning = int(item)

        for i in range(beginning,len(arr)): #iterate through files in array until bubble is selected 
            scan = cv2.imread('cv2op/' + str(i)+'cv2'+'.jpg', 0)
            cv2.imshow("scan", scan) #display picture
            cv2.setMouseCallback('scan', bubble_select) #apply click event
            cv2.waitKey()
            if bubble_loc != []: #break out of iteration once bubble is selected
                start = i #save starting index for bubble calculation
                bubs = True
                break
            elif i == beginning + 20:
                bubs = False
                break

        #####
        #CALCULATING CONTOURS
        #####
        areas = [] #empty array will hold areas
        bubbles = [] #empty array will hold contours
        bubbles_shapely = [] #empty array will hold shapely contours

        if bubs:
            #we will be deconvoluting or otherwise cleaning the images as we go to improve contour selection
            for i in range(start, len(arr)): #iterate through all scans from SELECTED To end
                #from shaan's code for pre-contour recognition thresholding
                scanz = cv2.imread('cv2op/' + str(i)+'cv2'+'.jpg', cv2.IMREAD_COLOR)
                imagee = cv2.cvtColor(scanz, cv2.COLOR_BGR2GRAY)
                '''cv2.imshow("scan", imagee) #display picture
                cv2.waitKey()
                imagee = cv2.bitwise_not(imagee)
                cv2.imshow("scan", imagee) #display picture
                cv2.waitKey()'''
                imagee = cv2.fastNlMeansDenoising(imagee,imagee,5,5,20)
                ret3,imagee = cv2.threshold(imagee,0,255,cv2.THRESH_OTSU)
                #cv2.imshow("scan", imagee) #display picture
                #cv2.waitKey()
    
    
                #contour detection, from shaan's contour detection code
                #also, for the record, I hate openCV with a passion right now
                #like, why on earth sdoes imshow only work SOMETIMES
                #AND WHY IS THIS LIBRARY SO BROKEN FOR LINUX I SWEAR
                contours, hierarchy = cv2.findContours(imagee, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                  
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
                    
                    #find the contour of SMALLEST AREA that contains the coord
                    asmallest = 999999999 #placeholder
                    contour_bubble = None #placeholder
                    contour_shapely = None #placeholder
                    for cnt in contours:
                        cnt = cnt
                        if len(cnt) >= 4: #lines cannot become polygons
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
                                if a < asmallest + 30:
                                    #conditions met? take contour
                                    asmallest = a
                                    contour_bubble = cnt
                                    contour_shapely = polygon

                    #store chosen contour
                    areas.append(asmallest)
                    original = asmallest
                    bubbles.append(contour_bubble)  
                    bubbles_shapely.append(contour_shapely)
                    with open('./areas.csv', 'a') as f: #save results
                        writer = csv.writer(f)
                        line = [arr[i], asmallest, asmallest*pix_width*pix_width]
                        writer.writerow(line)
        
                else:
                    #for remaining images, this is based on previous coordinates
                    asmallest = original #placeholder
                    contour_bubble = None #placeholder
                    contour_shapely = None #placeholder
                    prev = bubbles_shapely[-1]
                    
                    #we want the search to end if there is no bubble
                    bubble_here = False
        
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
                            if polygon.exterior.distance(loc) < 15:
                                #see if overlap with previous contour
                                if True:
                                    #see if area is smallest
                                    if (asmallest - 500) < a < (asmallest+500):            
                                        asmallest = a
                                        contour_bubble = cnt
                                        contour_shapely = polygon
                                        bubble_here = True
        
                    #break out of the loop if the chosen contour is stupidly large ie: not a bubble                    
                    if asmallest >= 10*max(areas):
                        bubble_here = False
                           
                    if bubble_here:           
                        areas.append(asmallest)  
                        bubbles.append(contour_bubble)
                        bubbles_shapely.append(contour_shapely)
                        with open('./areas.csv', 'a') as f: #save results
                            writer = csv.writer(f)
                            line = [arr[i], asmallest, asmallest*pix_width*pix_width]       
                            writer.writerow(line)     
                    else:
                        break
    
                if show: #if they WANT to review, show them each scan for vetting
                    first, second = contour_shapely.exterior.xy
                    plt.plot(first, second, zorder = 1000000, c = 'g', linewidth = 1)
                    plt.imshow(scan, zorder = 1)
                    name = str(i)+"_selected.png"
                    plt.axis("off")
                    plt.savefig(name)
                    plt.show()
                else: #regardless, draw and save the image
                    first, second = contour_shapely.exterior.xy
                    plt.plot(first, second, zorder = 1000000, c = 'g', linewidth = 1)
                    plt.imshow(scan, zorder = 1)
                    plt.axis("off")        
                    name = str(i)+"_selected.png"
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
            endv += volume
            
            
    root = tk.Tk() 
    root.withdraw()
    font1 = font.Font(name='TkCaptionFont', exists=True)
    font1.config(family='Calibri', size=15)
    messagebox.showinfo(message="The bubble volume is "+str(endv)+"mm^3")

iterate([200, 250])
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

