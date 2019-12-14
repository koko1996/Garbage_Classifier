###################################################################################################
# EECS4422 Helper Functions                                                                       #
# Filename: preprocess.py                                                                         #
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
# Student Number: 215168057                                                                       #
###################################################################################################


import os
import cv2
import time
import math
import numpy as np

def switch_colors(images_path):
    '''
    Descirption: switch_colors is a helper function to flip the colors of binary masked images
    and save the fliped images (all images in the image_path directory) at the same location 
    as the given image with the same name as the original image appended the suffix 'b.jpg'
    Parameters: 
        - image_path: path to the images
    returns: 
    '''
    jpg_extension = '.jpg'
    slash = '/'

    for filename in os.listdir(images_path):
        if filename.endswith(jpg_extension): 
            print(images_path+filename)
            img = cv2.imread(images_path+filename,0)
            cv2.imshow('Image '+filename, img)
            img[img == 0] =255
            img[img > 1] = 0
            img[img == 1] = 255
            cv2.imshow('Image '+filename, img)
            cv2.imwrite(images_path+filename+'b.jpg', img) 

def rotate_images(images_path):
    '''
    Descirption: rotate_images is a helper function to rotate all images in the images_path
    where it rotates each image 5 degrees 36 times and saves the rotated images
    Parameters: 
        - image_path: path to the images
    returns: N/A
    '''
    jpg_extension = '.jpg'
    slash = '/'

    for filename in os.listdir(images_path):
        if filename.endswith(jpg_extension): 
            print(images_path+filename)
            img = cv2.imread(images_path+filename,1)
            num_rows, num_cols = img.shape[:2]

            for angle in range(0,180,5):
                # rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
                rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
                img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows), borderValue=(255,255,255))
                # cv2.imshow(str(angle)+' bImage '+filename, img_rotation)
                cv2.imwrite(images_path+'_'+filename+str(angle)+'.jpg', img_rotation) 
            

def mask_images(images_path):
    '''
    Descirption: mask_images is a helper function to mask all the images that are in the 
    images_path using the mask for each image.
    Parameters: 
        - image_path: path to the images
    returns: N/A
    '''
    jpg_extension = '.jpg'
    png_extension = '.png'
    slash = '/'

    for filename in os.listdir(images_path):
        if filename.endswith(jpg_extension): 
            image_name = filename[:-len(jpg_extension)]
            img = cv2.imread(images_path+image_name+jpg_extension,1)
            m_img = cv2.imread(images_path+image_name+png_extension,0)

            m_img[m_img > 0] = True
            img = cv2.resize(img, (800, 600))
            m_img = cv2.resize(m_img, (800, 600))
            merge = cv2.bitwise_and(img,img,mask=m_img)

            m_inverse = m_img.copy()
            m_inverse[m_inverse == 0] = 255
            m_inverse[m_inverse == 1] = 0
            m_inverse_3d=  np.dstack([m_inverse,m_inverse,m_inverse])
            merge = cv2.bitwise_or(merge,m_inverse_3d,dst =merge, mask=m_inverse)    
            cv2.imwrite(images_path+filename+'b.jpg', merge) 

if __name__ == "__main__":
    # switch_colors('Images/box_juice/')
    # rotate_images('Images/can_rotten/')
    # rotate_images('Images/can_tomato/')
    # rotate_images('Images/can_chowder/')
    # mask_images('Images/box_juice/')
    # rotate_images('Images/bulb/')
    # switch_colors('Images/Mine/')
    # replicate_images('Images/can_chowder/')

    cv2.waitKey(0)
    cv2.destroyAllWindows()