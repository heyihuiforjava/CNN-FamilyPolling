# @title Import, install and authorisations
# Install the PyDrive wrapper & import libraries.
import time
from os import listdir

import keras
# importing the Networks related lib
import numpy as np
from scipy.io import loadmat


def family_from_HDD(category):
    family_dict = {}
    file_list = [f for f in listdir(folder_of_saved_model) if file_of_saved_model in f]
    for file in file_list:
        file = folder_of_saved_model + file
        if "cat_" + str(category) in file:
            modelNo = file[-10] + file[-4]
            print(file)
            family_dict[str(modelNo)] = keras.models.load_model(file)
            #   print(family_dict[str(modelNo)])
    print("len(family_dict) ", len(family_dict))
    return family_dict


def majority_polling_bin(family_dict, x_test, verbose=0, batch_size=None):
    try:
        #         predict_list = {}
        #         print("len(family_dict) ",len(family_dict))
        votes = np.zeros((len(x_test), 1))  # before voting, all votes are 0

        for model in family_dict:
            prediction = family_dict[str(model)].predict(x_test, batch_size=batch_size, verbose=verbose)
            prediction = np.round(prediction, decimals=0)  # round predictions to 0 or 1
            #             print("prediction", prediction, np.shape(prediction), type(prediction))
            votes += prediction
        votes = votes / len(family_dict)
        #         print("votes", np.round( votes , decimals=0 ), np.shape(votes), type(votes))
        return np.round(votes, decimals=0)  # round votes to 0 or 1

    except Exception as e:
        import traceback, sys
        print(e, file=sys.stderr)
        traceback.print_exc()


def majority_polling_global(x_test, y_test=None, verbose=0, batch_size=None):
    """
    Have the family of CNN stored on HDD make a guess about he classification of each images in x_train.
    If y_test is provided, then the function return the accuracy of family guess.
    If not, then return a matrix of shape (len(x_test, 6) containing the family guess for the images

    :param x_test: matrix of shape (numberOfImages, imageSize, imageSize, 1) containing the image dataset
    :param y_test: (optional) matrix of shape (numberOfImages, 6) containing the output dataset
    :param verbose:
    :param batch_size:
    :return: If y_test is provided : the accuracy of the family guess, as a float between 0 ant 1. If not, the matrix of family guesses.

    """
    # Having the family vote over the x_test dataset
    votes = [[] for _ in x_test]

    for category in range(6):
        family_dict = family_from_HDD(category)
        tmp = majority_polling_bin(family_dict, x_test, verbose=0, batch_size=None)
        del family_dict
        for picture in range(len(votes)):
            votes[picture].append(tmp[picture][0])
    if y_test is None:
        return votes
    else:  # if y_test provided
        # # Evaluating the family accuracy
        False_no = 0
        for i in range(len(y_test)):
            if False in list(y_test[i] == votes[i]):
                False_no += 1
        return 1 - (False_no / len(y_test))


if __name__ == "__main__":
    t = time.time()
    # variables
    model_to_use = "Deepdog"  # must be one of the name in CNN_global() dict
    folder_of_saved_model = "models\\"
    file_of_saved_model = "my_%s_model"%model_to_use
    nb_training_session = 1
    epoch_by_training_session = 1
    img_rows = img_cols = image_size = 128
    output_number = 1
    input_shape = (img_rows, img_cols, 1)  # image image_size*image_size greyscale

    # load the dataset from GDrive to VM
    dataset_file = 'datasets\\Training_Images_%d_mean.mat' % image_size
    print(dataset_file)

    # # Evaluating the family
    # Load test dataset
    mat = loadmat(dataset_file)
    x_test = mat['x_test']
    y_test = mat['y_test']
    del mat
    # normalize data
    x_test = x_test.astype('float32') / 255.0
    # # reshape data due to a mistake in storage
    x_test = np.rot90(np.rot90(x_test, axes=(2, 0)), axes=(1, 2))
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # avoid float outputs
    y_test[y_test != 1] = 0

    # reduction for test
    y_test = y_test[:10]
    x_test = x_test[:10]

    # Having the family vote over the x_test dataset
    print(majority_polling_global(x_test=x_test,y_test=y_test))
    print("time elapsed", time.time() - t)
