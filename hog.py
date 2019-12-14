###################################################################################################
# EECS4422 Garbage Classifier                                                                     #
# Filename: hog.py                                                                                #
# Author: NANAH JI, KOKO                                                                          #
# Email: koko96@my.yorku.com                                                                      #
# Student Number: 215168057                                                                       #
###################################################################################################

import os
import cv2
import random
import imutils
import math as m
import numpy as np
from pprint import pprint
from sklearn import svm
from imutils import paths
from skimage import feature
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def correct_prediciton(prediction, ground_truth):
    '''
    Descirption: correct_prediciton is a helper function returns true if the prediction
    is in the correct bin (Organic, recyclable, non_recyclable)
    Parameters: 
        - prediction: string representing the prediction label (such as apple, banana)
        - ground_truth: string representing the ground_truth label (such as apple, banana)
    returns: true if the prediction is in the correct bin ["Organic","Recyclable","Non-Recyclable"]
    and false otherwise
    '''
    organic = ['apple','apple_rotten','banana','banana_rotten']
    recyclable = ['box_cartoon','box_juice','can_chowder','can_rotten','can_soda','can_tomato']
    non_recyclable = ['bulb']

    ans = False
    if prediction in organic and ground_truth in organic:
        ans = True
    elif prediction in recyclable and ground_truth in recyclable:
        ans = True
    elif prediction in non_recyclable and ground_truth in non_recyclable:
        ans = True

    return ans



def train(train_info):
    '''
    Descirption: train is a helper function that train a model given training data. The training 
    algorithm is support vector machine
    Parameters: train_info that contain information of the training data
        - train_info: list of tuples containing the path to the image as the first element in the 
        tuple and the ground truth label of that image as the second element in the tuple
    returns: model that is trained on the training info that was passed as an argument
    '''
    data = []
    labels = []    
    
    print("[INFO] Starting to generate the descriptors...")
    for path, label in train_info:
        # read the image based on it's path
        img = cv2.imread(path,0)
        # resize the image for hog to work
        img = cv2.resize(img, (128, 64))
        # extract Histogram of Oriented Gradients from the image
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",	visualize=True)

        # Uncomment the next section to visualize the HOG image
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # hogImage = cv2.resize(hogImage, (400, 300))
        # cv2.imshow("HOG Image "+str(i), hogImage)
    
        # update the data and labels
        data.append(H)
        labels.append(label)

    print("[INFO] Training classifier...")

    # KNN
    # model = KNeighborsClassifier(n_neighbors=1)
    # model.fit(data, labels)

    # Create the svm Classifier
    model = svm.SVC(kernel='linear') # Linear Kernel
    # Train the svm Classifier
    model.fit(data, labels)

    print("[INFO] Training done...")

    return model


def test(test_info,model):
    '''
    Descirption: test is a function that tests a given model on the given testing data.
    Parameters: test_info that contain information of the testing data
        - test_info: list of tuples containing the path to the image as the first element in the 
        tuple and the ground truth label of that image as the second element in the tuple
    returns: integer that represents the number of misclassifications done by the model on the
    test data.
    '''

    number_of_failures = 0                    # variable to accumulate the number of misclassifications
    errors = []
    for (i, info) in enumerate(test_info):
        # read the image based on it's path
        image = cv2.imread(info[0],0)
        # resize the image so that hog works
        image = cv2.resize(image, (128, 64))

        # extract Histogram of Oriented Gradients from the test image and predict
        (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
        pred = model.predict(H.reshape(1, -1))[0]

        # Uncomment the next part to visualize the HOG image
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
        
        # check the predicted bin
        if not correct_prediciton(pred,info[1]):
            number_of_failures += 1
            errors.append((pred,info[1]))

            # Uncomment the next part to visualize only the images that are misclassified
            # print("failed: " + info)
            # draw the prediction on the test image and display it
            # cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            # cv2.imshow("Test Image #{}".format(i + 1), image)
        
    return (number_of_failures,errors)


def retrieve_paths_per_label(label,cardinality):
    '''
    Descirption: retrieve_paths_per_label is a helper function that retieves cardinality
    number of paths of images that belong to the given label. 
    Parameters:
        - label: strings represents the label of the images to retireve (such as apple, banana)
        - cardinality: specifies the number of paths to retrieve for the given label
    returns: list of size cardinality containing strings where each string represents the
    path in operating system of an image that has the given label (such as the paths of all apple images)
    '''
    images=[]
    images_path = 'Images/'+label
    jpg_extension = '.jpg'
    slash = '/'
    for filename in os.listdir(images_path)[:cardinality]:
        if filename.endswith(jpg_extension):
            images.append((images_path+slash+filename,label))
    return images

def retrieve_all_paths(labels,cardinality):
    '''
    Descirption: retrieve_all_paths is a helper function that retieves cardinality
    number of paths of images that belong to each category specified in the labels list
    Parameters:
        - labels: list of strings containing the names of each of the categories
        - cardinality: specifies the number of paths to retrieve per category
    returns: dictonary which has label as the key and a list as the value where the list 
    contains cardinality number of strings where each string represents the path in the 
    operating system of an image in the given label (key is the given label for that list)
    '''
    images={}
    for label in labels:
        images[label] = retrieve_paths_per_label(label,cardinality)
    return images

def evaluate_model(dataset,input_size,chunk_size):
    '''
    Descirption: evaluate_model is a function that evaluates the model on the given testing data 
    using cross validation as the evaluation method.
    Parameters: 
        - dataset: dictonary of label as the key and list of tuples containing the path to the image 
        as the value
        - input_size: number images to use for each category (label)
        - chunck_size: The size of each chunk for cross validation
    returns: integer that represents the number of misclassifications done by the model on the complete
    dataset
    '''

    # shuffle the given data
    for label, paths in dataset.items():
        random.shuffle(dataset[label])

    total_errors = 0      # variable to accumulate the number of errors
    total_tests = 0
    error_types = []
    print("[INFO] Starting evaluation...")
    for j in range(1, int(input_size/chunk_size)):       
        # devide the data to test and train sets 
        train_data = []
        test_data = []
        for label in dataset:
            train_data += dataset[label][: (chunk_size * (j-1))]+ dataset[label][ (chunk_size * j):]
            test_data  += dataset[label][ (chunk_size * (j-1)):(chunk_size * j)]
        model = train(train_data)
        count, errors = test(test_data,model)
        error_types = error_types + errors
        total_errors += count
        total_tests += len(test_data)
    print("[INFO] Finished evaluation... total_tests:",total_tests)
    return (total_errors,error_types)

def classify_garbage(input_image, model):
    '''
    Descirption: classify_garbage is a helper function classifies an image based on a three-bin 
    garbage classificaation system. Returns the category the item in the image belongs to 
    Parameters: 
        - input_image: cv2 image object to be classified
        - model: the model to use for classification
    returns: string representing the category of the object which is one of the following
    ["Undefined","Organic","Recyclable","Non-Recyclable"]
    '''
    
    image = input_image.copy()
    image = cv2.resize(image, (128, 64))
    (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    label = model.predict(H.reshape(1, -1))[0]
    ans = "Undefined"
    if label in ['apple','apple_rotten','banana','banana_rotten']:
        ans = "Organic"
    elif label in ['box_cartoon','box_juice','can_chowder','can_rotten','can_soda','can_tomato']:
        ans = "Recyclable"
    elif label in ['bulb']:
        ans = "Non-Recyclable"
    return ans


def plot_misclassifications(errors):
    '''
    Descirption: plot_misclassifications is a helper function to plot the misclassifications
    Parameters: 
        - errors: dictonary that contains the errors and number of misclassifications
    returns: N/A
    '''
    complete_errors = {}

    for predicted, actual in errors:
        if actual not in complete_errors:
            complete_errors[actual] = {predicted:1}
        else:
            prev_value = complete_errors[actual]
            if predicted not in prev_value:
                prev_value[predicted] = 1
            else:
                prev_value[predicted] += 1
    
    for actual, miss_predicted in complete_errors.items():
        row = []
        height = []
        for predicted, count in miss_predicted.items():
            row.append(predicted)
            height.append(count)
        
        y_pos = np.arange(len(row))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, row)
        plt.title("Ground Truth Lable: " + actual)
        plt.xlabel('Incorrectly Predicted Classes')
        plt.ylabel('Number of Miclassifications')
        plt.savefig('Misclassifications_'+str(actual)+'.png')
        plt.clf()
    pprint (complete_errors)


if __name__ == "__main__":
    INPUT_SIZE = 150
    CROSS_CORRELATION_CARDINALITY =  15
    print("[INFO] Starting to read the data...")
    labels = ['apple','apple_rotten','banana','banana_rotten','box_cartoon','box_juice','bulb','can_chowder','can_rotten','can_soda','can_tomato']
    complete_data = retrieve_all_paths(labels,INPUT_SIZE)
    print("[INFO] Done reading the data...")


    # Evaluate the model using cross validation
    count_errors,errors = evaluate_model(complete_data, INPUT_SIZE, CROSS_CORRELATION_CARDINALITY)
    print ("Result of Cross correlation", count_errors)
    print ("Errors", errors)
    plot_misclassifications(errors)

    # Main part of the code to actually train and use the model
    # train_data = []
    # for label in complete_data:
    #     train_data += complete_data[label]
    # print(train_data)
    
    # model = train(train_data)
    # img = cv2.imread('Images/apple_rotten/apple_rotten_21.jpg',0)
    # cv2.imshow("Input image", img)
    # print(classify_garbage(img,model))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
