import cv2
import json
import requests
import numpy as np
import argparse
import imutils
import glob
import math

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 2

def openJson(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data

def crop(img):
    return img[0:64,0:64]

def crop1(img, x1, x2, y1, y2):
    return img[x1:x2,y1:y2]

def main():
    
    path = 'test_images/ingame.png'
    data = openJson("data.json")
    data = get_item_images(data)
    image = cv2.imread(path)
    
    h, w, _ = image.shape
    image = crop1(image, math.floor(h*0.30), math.floor(h*0.65), math.floor(w*0.37), math.floor(w*0.8))
    count = 0
    for item in data:
        template = cv2.Canny(data[item]["image"], 50, 200)
        #template = data[item]['image']
        image = find(template, image, data[item]['id'])



    cv2.imshow("Image", image)
    cv2.setMouseCallback('Image', click_event) 
    cv2.waitKey(0)

def get_item_images(data):
    for img_id in data:
        data[''+img_id+'']['image'] = crop(cv2.imread('images/'+img_id+'.png', 0))
    return data

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  

def find(template, image, image_id):
    found = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (tH, tW) = template.shape[:2]
    for scale in np.linspace(1.80, 1.84, 1)[::-1]:

        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 50, 200)
        #edged = resized
        result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if False:
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2);
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
        
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    print(maxVal)
    (maxVal, maxLoc, r) = found
    if maxVal > 0.30:
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, image_id, (startX,endY), font, fontScale, fontColor, lineType)
    return image


main()