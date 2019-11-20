import os
import cv2
import time
import math
import numpy as np

def switch_colors(images_path ):
    jpg_extension = '.jpg'
    slash = '/'

    for filename in os.listdir(images_path):
        if filename.endswith(jpg_extension): 
            print(images_path+filename)
            img = cv2.imread(images_path+filename,0)
            cv2.imshow('Image '+filename, img)
            img[img == 0] =1
            img[img > 1] = 0
            img[img == 1] = 255
            cv2.imshow('Image '+filename, img)
            cv2.imwrite(images_path+'b'+filename, img) 

def rotate_images(images_path):
    jpg_extension = '.jpg'
    slash = '/'

    for filename in os.listdir(images_path):
        if filename.endswith(jpg_extension): 
            print(images_path+filename)
            img = cv2.imread(images_path+filename,1)
            num_rows, num_cols = img.shape[:2]

            for angle in range(0,180,10):
                rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
                img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows), borderValue=(255,255,255))
                # cv2.imshow(str(angle)+' bImage '+filename, img_rotation)
                cv2.imwrite(images_path+ str(angle)+'_'+filename, img_rotation) 
            

def mask_images(images_path):
    jpg_extension = '.jpg'
    png_extension = '.png'
    slash = '/'

    for filename in os.listdir(images_path):
        if filename.endswith(jpg_extension): 
            print(images_path+filename)
            image_name = filename[:-len(jpg_extension)]
            img = cv2.imread(images_path+image_name+jpg_extension,1)
            m_img = cv2.imread(images_path+image_name+png_extension,0)
            
            # cv2.imshow('Given Can',img)
            # cv2.imshow('Given Mask',m_img)

            m_img[m_img > 0] = True
            img = cv2.resize(img, (800, 600))
            m_img = cv2.resize(m_img, (800, 600))
            merge = cv2.bitwise_and(img,img,mask=m_img)
            # cv2.imshow('Masked ',merge)

            m_inverse = m_img.copy()
            m_inverse[m_inverse == 0] = 255
            m_inverse[m_inverse == 1] = 0
            m_inverse_3d=  np.dstack([m_inverse,m_inverse,m_inverse])
            merge = cv2.bitwise_or(merge,m_inverse_3d,dst =merge, mask=m_inverse)    
            # cv2.imshow('Give Mask Reverse',m_inverse)
            # cv2.imshow('Masked Reverse'+filename,merge)
            cv2.imwrite(images_path+'b'+filename, merge) 

if __name__ == "__main__":
    # switch_colors('Images/box_juice/')
    # rotate_images('Images/can_soda/')
    # rotate_images('Images/can_tomato/')
    # rotate_images('Images/can_chowder/')
    # mask_images('Images/box_juice/')
    # rotate_images('Images/box_cartoon/')
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()