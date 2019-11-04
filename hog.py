import cv2
import imutils
import math as m
import numpy as np
from imutils import paths
from skimage import feature
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



def train(train_info):
    data = []
    labels = []    
    i=0
    for path, label in train_info:
        # extract Histogram of Oriented Gradients from the logo
        img = cv2.imread(path,0)
        img = cv2.resize(img, (200, 100))
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",	visualize=True)
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image "+str(i), hogImage)

        # update the data and labels
        data.append(H)    
        labels.append(label)
        i=i+1

    print("[INFO] training classifier...")
    # KNN
    # model = KNeighborsClassifier(n_neighbors=1)
    # model.fit(data, labels)

    #Create a svm Classifier
    model = svm.SVC(kernel='linear') # Linear Kernel
    model.fit(data, labels)


    print("[INFO] training done...")
    return model


def test(test_info,model):
    failures = 0
    for (i, info) in enumerate(test_info):
        # load the test image, convert it to grayscale, and resize it to
        # the canonical size
        image = cv2.imread(info[0],0)
        image = cv2.resize(image, (200, 100))

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



def retrieve_images(label,cardinality):
    images=[]
    for i in range(cardinality):
        images.append(('images/'+label+'/'+label + '_' + str(i) +'.jpg',label))

    return images

if __name__ == "__main__":
    apple_data = (retrieve_images('apple',5)) 
    banana_data = (retrieve_images('banana',5)) 
    box_data = (retrieve_images('box',5)) 
    bulb_data = (retrieve_images('bulb',5)) 
    can_data = (retrieve_images('can',5))
    
    CHUNK_SIZE = 1
    total_errors=0      # variable to accumulate sum of square of test errors
    for j in range(1,6):       
        # devide the data to test and train sets 
        apple_train_data = apple_data[: (CHUNK_SIZE * (j-1))]+ apple_data[ (CHUNK_SIZE * j):]
        apple_test_data = apple_data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]
        banana_train_data = banana_data[: (CHUNK_SIZE * (j-1))]+ banana_data[ (CHUNK_SIZE * j):]
        banana_test_data = banana_data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]
        box_train_data = box_data[: (CHUNK_SIZE * (j-1))]+ box_data[ (CHUNK_SIZE * j):]
        box_test_data = box_data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]
        bulb_train_data = bulb_data[: (CHUNK_SIZE * (j-1))]+ bulb_data[ (CHUNK_SIZE * j):]
        bulb_test_data = bulb_data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]
        can_train_data = can_data[: (CHUNK_SIZE * (j-1))]+ can_data[ (CHUNK_SIZE * j):]
        can_test_data = can_data[ (CHUNK_SIZE * (j-1)):(CHUNK_SIZE * j)]


        train_data = apple_train_data + banana_train_data + box_train_data + bulb_train_data + can_train_data
        test_data = apple_test_data + banana_test_data + box_test_data + bulb_test_data + can_test_data
        model = train(train_data)
        total_errors += test(test_data,model)

    print ("Total Error")
    print (total_errors)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
