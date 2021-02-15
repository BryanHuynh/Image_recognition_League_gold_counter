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
lineType               = 1

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
    loc = []
    for item in data:
        template = cv2.Canny(data[item]["image"], 100, 200)
        img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(image,100,200)
        matches = scaled_find_template(edged, template)
        loc.append({'id':item, 'matches': matches, 'template': template})

    for i in loc:
        item_id = i['id']
        image = display_match(image, i['matches'], i['template'], item_id)

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
  

def match_template(edged, template, threshold=0.9):
    """
    Matches template image in a target grayscaled image
    """

    res = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
    #print(res)
    matches = np.where(res >= threshold)
    return matches


def scaled_find_template(img_grayscale, template, threshold=0.9, scales=[0.5, 0.7, 0.9]):
    for scale in np.linspace(0.5, 1, 10)[::-1]:
        scaled_template = cv2.resize(template, (0,0), fx=scale, fy=scale)
        matches = match_template(
            img_grayscale,
            scaled_template,
            threshold
        )
        if np.shape(matches)[1] >= 1:
            return matches
    return matches


def display_match(image, loc, template, image_id):
    w, h= template.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + 32, pt[1] + math.floor(h/2)), (0,0,255), 2)
        cv2.putText(image, image_id, (pt[0], pt[1] + 32), font, fontScale, fontColor, lineType)
    return image

main()