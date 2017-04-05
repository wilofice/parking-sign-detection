import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from create_features import extract_features_for_test_data

color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = False  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

(winW, winH) = (128, 128)  # the sliding window size
IMG_SIZE = 128
sizex = 528
sizey = 41
# test_data = np.load('test_data.npy') if we want to test on test images

LR = 1e-3
MODEL_NAME = 'parkingsign-{}-{}.model'.format(LR, '2conv-basic')
convnet = input_data(shape=[None, sizex, sizey, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('/home/genereux/PycharmProjects/parking/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

def tracer_cadre(img, coordinates, color=(0, 0, 255), thick=2):
    imcopy = np.copy(img)
    for (x, y) in coordinates:
        cv2.rectangle(imcopy, (x, y), (x + winW, y + winH), color, thick)
    return imcopy


def sliding_window(picture, stepsize, windowsize):
    # slide a window across the image
    for y in xrange(0, picture.shape[0], stepsize):
        for x in xrange(0, picture.shape[1], stepsize):
            yield (x, y, picture[y:y + windowsize[1], x:x + windowsize[0]])


def search_windows(picture, ourmodel):
    on_windows = []  # list to receive positive detection windows
    best = 0
    (xmax, ymax) = (0, 0)
    for (x, y, window) in sliding_window(picture, stepSize=8, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        feature = extract_features_for_test_data(window, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

        feature = np.reshape(feature, (sizex, sizey))
        feature = np.array([feature]).reshape(-1, sizex, sizey, 1)  # Scale extracted features to be fed to classifier
        prediction = ourmodel.predict(feature)  # Predict using your model
        print(prediction)

        if prediction[0][0] > prediction[0][1]:
            if prediction[0][0] > best:
                best = prediction[0][0]
                # (xmax,ymax) = (x,y)
            on_windows.append((x, y))
            print("found")

    print('best prediction = ', max)
    # on_windows.append((xmax,ymax))
    return on_windows


image = cv2.imread('imagetest2.jpg')
image_copy = np.copy(image)

hot_windows = search_windows(image, model)

image_parsed = tracer_cadre(image_copy, hot_windows, color=(255, 0, 0), thick=2)

plt.imshow(image_parsed)
plt.show()
