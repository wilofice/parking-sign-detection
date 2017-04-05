#parking sign detection with deep learning using convolutionnal neural networks with tflearn

create_features.py: contain the code to load data from the hard drive and extract feature from them. Train Data et Test Data                       will be saved respectively in new_train_data.npy and test_data.npy

create_model.py: contain the main code which create and train the model. The model is saved to parkingsign-0.001-2conv-                          basic.model.meta
                 The output of the training steps are in the file "training_steps.txt"

test_model.py: contain the code to test the model on 'one' image. (not all the test images contained in residential/img
                because it would take too many time to compute all the predictions)

The model have been tested on "imagetest1.jpg" and "imagetest2.jpg"; The results are images called "resultat(j)-imagetest(k).png". And Predictions for each sliding window for each test on each image are saved in "predictions-resultat(j)-imagetest(k).txt"
