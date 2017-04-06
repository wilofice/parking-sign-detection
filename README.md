#parking sign detection with deep learning using convolutionnal neural networks with tflearn

grab_data.py: contain the code to load data from the hard drive and extract feature from them. Train Data et Test Data  are saved respectively in new_train_data.npy and test_data.npy

train_model.py: contain the main code which implement the convolutional neural net and train the model. The model is saved to parkingsign-0.001-2conv-basic.model.meta
                 The output of the training steps are in the file "train_steps_log.txt". You can find there, the accuracy and the loss at the end of each epoch.

test_model-v1.py: contain the code to test the model on 'one' image. (not all the test images contained in residential/img
                because it would take too many time to compute all the predictions on each image)
               	Run this code to test the model on an image. The output is the image with all the sliding windows that may contain a parking sign.

test_model-v2.py: contain the code to test the model on 'one' image. (not all the test images contained in residential/img
                because it would take too many time to compute all the predictions on each image)
               	Run this code to test the model on an image. The output is the image with the sliding windows with the best probability to contain a parking sign.

The model have been tested on "input1.jpg", "input2.jpg", "input3.jpg". The results are images called "output". 

#TO TEST THE CODE

No need to load all the data and  train the model. The model that I have trained is already there and save in the file "parkingsign-0.001-2conv-basic.model.meta".
So to test the code, just run test_modelv1.py. Don't forget to change the location of the image you will be using to test. 

#TO REBUILD EVERYTHING FROM ZERO

-First grab training data. For that, uncomment the last line of grab_data.py in order to call the function get_train_data() and run the code. After that, recomment this line. 

-Secondly, train the model by running train_model.py

-Finally run test_model-v1.py to test the neural net on an image. Don't forget to change the location of the image you will be using to test. 