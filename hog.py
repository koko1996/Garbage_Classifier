import os
import cv2
import imutils
import math as m
import numpy as np
from sklearn import svm
from imutils import paths
from skimage import feature
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier



def train(train_info):
    data = []
    labels = []    
    i=0
    
    print("[INFO] Starting to generate the descriptors...")
    for path, label in train_info:
        # extract Histogram of Oriented Gradients from the logo
        img = cv2.imread(path,0)
        img = cv2.resize(img, (128, 64))
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",	visualize=True)
                
        # visualize the HOG image
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # hogImage = cv2.resize(hogImage, (400, 300))
        # cv2.imshow("HOG Image "+str(i), hogImage)
    
        # update the data and labels
        data.append(H)
        labels.append(label)
        i=i+1

    print("[INFO] Training classifier...")
    # KNN
    # model = KNeighborsClassifier(n_neighbors=1)
    # model.fit(data, labels)

    # Create a svm Classifier
    model = svm.SVC(kernel='linear') # Linear Kernel
    model.fit(data, labels)


    print("[INFO] Training done...")
    return model


def test(test_info,model):
    failures = 0
    for (i, info) in enumerate(test_info):
        # load the test image, convert it to grayscale, and resize it to
        # the canonical size
        image = cv2.imread(info[0],0)
        image = cv2.resize(image, (128, 64))

        # extract Histogram of Oriented Gradients from the test image and
        # predict the make of the car
        (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
        pred = model.predict(H.reshape(1, -1))[0]

        # visualize the HOG image
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
        if pred != info[1]:
            failures += 1
            print("failed: ")
            print(info)
            # draw the prediction on the test image and display it
            cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("Test Image #{}".format(i + 1), image)
        
    return failures


def retrieve_images_per_label(label,cardinality):
    images=[]
    images_path = 'Images/'+label
    jpg_extension = '.jpg'
    slash = '/'
    for filename in os.listdir(images_path)[:cardinality]:
        if filename.endswith(jpg_extension):
            images.append((images_path+slash+filename,label))

    return images

def retrieve_all_images(labels,cardinality):
    images={}
    for label in labels:
        images[label] = retrieve_images_per_label(label,cardinality)
    return images

def evaluate_model(dataset,chunk_size):
    total_errors=0      # variable to accumulate sum of square of test errors
    
    print("[INFO] Starting evaluation...")
    for j in range(1,chunk_size):       
        # devide the data to test and train sets 
        train_data = []
        test_data = []
        for label in dataset:
            train_data += dataset[label][: (chunk_size * (j-1))]+ dataset[label][ (chunk_size * j):]
            test_data  += dataset[label][ (chunk_size * (j-1)):(chunk_size * j)]
        model = train(train_data)
        total_errors += test(test_data,model)
    print("[INFO] Finished evaluation...")
    return total_errors

def classify_garbage(input_image, model):
    image = input_image.copy()
    image = cv2.resize(image, (128, 64))
    (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    label = model.predict(H.reshape(1, -1))[0]
    ans = "Undefined"
    if label in ['apple','apple_rotten','banana','banana_rotten']:
        ans = "Organic"
    elif label in ['box_cartoon','box_juice','can_chowder','can_soda','can_tomato']:
        ans = "Recyclable"
    elif label in ['bulb']:
        ans = "Non-Recyclable"
    return ans


if __name__ == "__main__":
    INPUT_SIZE = 36
    CROSS_CORRELATION_CARDINALITY =  6

    print("[INFO] Starting to read the data...")
    labels = ['apple','apple_rotten','banana','banana_rotten','box_cartoon','box_juice','bulb','can_chowder','can_soda','can_tomato']
    complete_data = retrieve_all_images(labels,INPUT_SIZE)
    print("[INFO] Done reading the data...")

    # print ("Result of Cross correlation", evaluate_model(complete_data, CROSS_CORRELATION_CARDINALITY))
    
    train_data = []
    for label in complete_data:
        train_data += complete_data[label]
    # print(train_data)

    # Live Demo
    model = train(train_data)
    img = cv2.imread('Images/apple_rotten/apple_rotten_21.jpg',0)
    cv2.imshow("Input image", img)
    print(classify_garbage(img,model))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
