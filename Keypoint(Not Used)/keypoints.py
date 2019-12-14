import cv2
import math as m
import numpy as np


# Code to test Keypoints (Not used in the project)

def getKeypionts(img):
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return(kp, des)

def appleOneMask(img):  
    img[img <30] = 255
    img[img <245] = 0
    kernsize = 5 # the kernel size (should be an odd integer); larger values apply the filter over larger contexts
    img = cv2.medianBlur(img, kernsize) # smooth the image with the filter kernel
    return img    

def appleOne():
    apple_img = cv2.imread('images/test/Apple/apple_70.jpg',0)
    cv2.imshow('Given Apple',apple_img)

    apple_img = appleOneMask(apple_img)
    cv2.imshow('Given Mask',apple_img)
    kp, des = getKeypionts(apple_img)

    return (apple_img, kp, des)
    
def appleTwoMask(img):  
    img[img <30] = 255
    img[img <254] = 0
    kernsize = 5 # the kernel size (should be an odd integer); larger values apply the filter over larger contexts
    img = cv2.medianBlur(img, kernsize) # smooth the image with the filter kernel
    return img    

def appleTwo():
    apple_img = cv2.imread('Images/apple/apple_0.jpg',0)
    apple_img = cv2.resize(apple_img, (400, 300))
    cv2.imshow('Given Apple',apple_img)

    # apple_img = appleTwoMask(apple_img)
    # cv2.imshow('Given Mask',apple_img)
    kp, des = getKeypionts(apple_img)

    return (apple_img, kp, des)    

def appleThreeMask(img):  
    img[img <10] = 255
    img[img <220] = 0
    kernsize = 5 # the kernel size (should be an odd integer); larger values apply the filter over larger contexts
    img = cv2.medianBlur(img, kernsize) # smooth the image with the filter kernel

    return img    

def appleThree():
    apple_img = cv2.imread('Images/banana_rotten/banana_rotten_10.jpg',0)
    apple_img = cv2.resize(apple_img, (400, 300))
    # apple_img=apple_img[:-50, :]    
    # cv2.imshow('Given Apple',apple_img)

    # apple_img = appleThreeMask(apple_img)
    # cv2.imshow('Given Mask',apple_img)
    kp, des = getKeypionts(apple_img)

    return (apple_img, kp, des)   


def findMatchingKeypoints(apple_one,kp_apple_one,des_apple_one,apple_two,kp_apple_two,des_apple_two):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.array(des_apple_one,np.float32), np.array(des_apple_two,np.float32), k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    print(len(good_matches))
    left_matrix =[]
    right_matrix = []
    for m in good_matches:
        left_index = m.queryIdx
        right_index = m.trainIdx
        left_points = kp_apple_one[left_index].pt
        right_points = kp_apple_two[right_index].pt
        left_matrix.append(np.array(left_points))
        right_matrix.append(np.array(right_points))

    left_matrix=np.array(left_matrix,np.float32)
    right_matrix=np.array(right_matrix,np.float32)

    img_matches = np.empty((max(apple_one.shape[0], apple_two.shape[0]), apple_one.shape[1]+ apple_two.shape[1], 3), dtype=np.uint8)
    img_mtch = cv2.drawMatches(apple_one, kp_apple_one, apple_two, kp_apple_two, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches',img_mtch)
    return good_matches


if __name__ == "__main__":
    # apple_one, kp_apple_one, des_apple_one = appleOne()
    # apple_one_kp = cv2.drawKeypoints(apple_one,kp_apple_one,color=(0,255,0), outImage=np.array([]), flags=0)
    # cv2.imshow('Apple one keypoints',apple_one_kp)
    
    apple_two, kp_apple_two, des_apple_two = appleTwo()
    apple_two_kp = cv2.drawKeypoints(apple_two,kp_apple_two,color=(0,255,0), outImage=np.array([]), flags=0)
    cv2.imshow('Apple two keypoints',apple_two_kp)

    apple_three, kp_apple_three, des_apple_three = appleThree()
    apple_three_kp = cv2.drawKeypoints(apple_three,kp_apple_three,color=(0,255,0), outImage=np.array([]), flags=0)
    cv2.imshow('Apple three keypoints',apple_three_kp)

    findMatchingKeypoints(apple_two,kp_apple_two,des_apple_two,apple_three,kp_apple_three,des_apple_three)

    cv2.waitKey(0)
    cv2.destroyAllWindows()