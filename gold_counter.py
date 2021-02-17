import cv2
import json
import requests
import numpy as np
import argparse
import imutils
import glob
import math
import sys

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1


def openJson(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data 

def main(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1920, 1080))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (3,3), 0)
    image_edge = cv2.Canny(image_blur, 10, 200)


    item_data = openJson('data.json')
    item_data = get_item_images(item_data)

    item_data['3330']['image'] = cv2.resize(item_data['3330']['image'], (64,64))


    boxes = get_player_boxes(image, item_data)
    draw_player_boxes(image, boxes)
    player_box_images = get_player_box_images(image, boxes)

    players_items = []
    for box in player_box_images:
        items_locations = []
        for _id in item_data:
            #[(location, item_id)]
            items_locations.append((templateMatching(item_data[_id]['image'], box), _id))
        display_matches_w_id(items_locations, box)
        player_items = {"image": box, "items": items_locations}
        players_items.append(player_items)
    
    cost_images = []
    for player_items in players_items:
        cost_images.append(extend_and_add_cost(player_items, item_data))
        #showImage(player_items['image'])

    combined = make_rows_into_single(cost_images)
    #showImage(combined)

    cv2.imwrite("output.jpg", combined)

def extend_and_add_cost(player_items, item_data):
    items = extractItems(player_items['items'])
    cost = get_total_cost(items, item_data)
    extend = cv2.copyMakeBorder(player_items['image'], 5, 5, 0, 200, cv2.BORDER_CONSTANT)
    cv2.putText(extend, str(cost), (32*8, 32) , font, 1, fontColor, lineType)
    return extend



def get_player_box_images(image, boxes):
    images = []
    added_boxes = []
    for box in boxes:
        if not is_close_duplicates_from_list(box, added_boxes, 5):
            image_box = image[box[0][1]:box[1][1],box[0][0]:box[1][0]]
            images.append(image_box)
            added_boxes.append(box)
        #showImage(image_box)
        #print(box)

    return images

def is_close_duplicates_from_list(test, items, range):
    if len(items) == 0: return False
    for item in items:
        if test[0][0] + range >= item[0][0] and test[0][0] - range <= item[0][0]:
            if test[0][1] + range >= item[0][1] and test[0][1] - range <= item[0][1]:
                if test[1][0] + range >= item[1][0] and test[1][0] - range <= item[1][0]:
                    if test[1][1] + range >= item[1][1] and test[1][1] - range <= item[1][1]:
                        return True
    
    return False






def templateMatching(template, image, threshold = 0.5):
    template = cv2.resize(template, (32,32))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_edge = cv2.Canny(template_gray, 50,200)
    img_edge = cv2.Canny(img_gray, 50,200)

    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(img_edge, template_edge, cv2.TM_CCORR_NORMED)

    loc = np.where(res >= threshold)
    return loc

def get_player_boxes(image, item_data):
    farsight_loc, oracle_loc, ward_loc = findWards(image, item_data)
    w, h = (32,32)
    padding = 5
    item_row_size = (32 * 6) + 20
    boxes = []
    for pt in zip(*ward_loc[::-1]):
        box = ((pt[0] - item_row_size, pt[1] - padding), (pt[0] + w, pt[1] + h + padding))
        boxes.append(box)
        
    for pt in zip(*oracle_loc[::-1]):
        box = ((pt[0] - item_row_size, pt[1] - padding), (pt[0] + w, pt[1] + h + padding))
        boxes.append(box)
    
    for pt in zip(*farsight_loc[::-1]):
        box = ((pt[0] - item_row_size, pt[1] - padding), (pt[0] + w, pt[1] + h + padding))
        boxes.append(box)
    return boxes

def draw_player_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, box[0], box[1], (0,0,255), 2)

def findWards(image, item_data):
    template = item_data['3340']['image']
    stealthWard = templateMatching(template, image, threshold=0.52)
    template = item_data['3364']['image']
    oracle = templateMatching(template, image, threshold=0.52)
    template = item_data['3363']['image']
    farsight = templateMatching(template, image, threshold=0.52)
    return (farsight, oracle, stealthWard)




def display_matches(locs, image, image_id):
    w, h = (32,32)
    for loc in locs:
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv2.putText(image, image_id, (pt[0], pt[1] + 32), font, fontScale, fontColor, lineType)
        showImage(image)

    
def display_matches_w_id(locs, image):
    w, h = (32,32)
    for loc_id in locs:
        loc = loc_id[0]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv2.putText(image, loc_id[1], (pt[0], pt[1] + 32), font, fontScale, fontColor, lineType)



def extractItems(locs):
    items_x_postions = []
    items = []
    for loc_id in locs:
        loc = loc_id[0]
        for pt in zip(*loc[::-1]):
            if pt[1] > 0 or pt[0] > 0:
                # check if there is another value in locs with the same position
                # store the x value in a list. 
                # check if there exist another x value around 32 pixels

                if(not item_in_list(pt[0], items_x_postions, 10)):
                    items_x_postions.append(pt[0])
                    items.append(loc_id[1])
    #print(items)
    return items



def get_total_cost(items, items_data):
    cost = 0
    for item in items:
        #print(items_data[item]['name'])
        cost += items_data[item]['gold']
    return cost

def item_in_list(item, items, range):
    if len(items) == 0: return False
    n_item = item - range
    p_item = item + range
    for i in items:
        if n_item <= i and p_item >= i:
            return True 
    return False        

def showImage(image, name='image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)

def showItems(item_data, canny = False, gray = False):
    '''
    shows all items in a grid 
    '''
    item_images = []
    for item in item_data:
        item_image = item_data[item]['image']
        item_images.append(item_image)

    grid = turn_images_to_grid(item_images)
    if gray or canny:
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    if canny:
        grid = cv2.GaussianBlur(grid, (3,3), 0)
        grid = auto_canny(grid)

    cv2.imshow('grid', grid)
    cv2.waitKey(0)



def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def turn_images_to_grid(images, rows = 15):
    '''
    puts all images into a grid format
    '''
    size = 32
    length = len(images)
    per_row = math.floor(length / rows)
    image_rows = []
    

    for i in range(0, rows):
        row = np.hstack(images[i * per_row : (per_row * i) + per_row])
        image_rows.append(row)

    last_row = np.hstack(images[rows * per_row:])
    image_rows.append(last_row)

    return make_rows_into_single(image_rows)

def make_rows_into_single(img_list):
    padding = 2
    max_width = []
    max_height = 0
    for img in img_list:
        max_width.append(img.shape[1])
        max_height += img.shape[0]
    w = np.max(max_width)
    h = max_height + padding
    #print(w,h)

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((h, w, 3), dtype=np.uint8)
    #print(final_image.shape)
    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
        current_y += image.shape[0]
    return final_image





def get_item_images(data):
    for img_id in data:
        data[''+img_id+'']['image'] = cv2.imread('images/'+img_id+'.png')
    return data


    cv2.imshow('wide', image_edge2)
    cv2.waitKey(0)



