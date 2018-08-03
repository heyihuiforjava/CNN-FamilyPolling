import pickle
import time

import numpy as np
from keras import backend as K
from scipy.io import loadmat

# variables
from CNN_generation import CNN_global


def stopTraning(score_liste):
    try:
        return (score_liste[-1][1] <= score_liste[-2][1] and score_liste[-2][1] <= score_liste[-3][1] and
                score_liste[-3][1] <= score_liste[-4][1])
    except IndexError:  # si la liste n'a pas assez d'element, au debut typiquement
        return False
    except:
        return False


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, 0)


# variables
model_to_use = "Deepdog"  # must be one of the name in CNN_global() dict
folder_to_save_model = "models\\"
file_to_save_model = "BSize_Analysis_my_%s_model" % model_to_use
img_rows = img_cols = image_size = 128
output_number = 6
input_shape = (img_rows, img_cols, 1)  # image image_size*image_size greyscale

# load the dataset from Disk
dataset_file = '..\\datasets\\Training_Images_%d_mean.mat' % image_size
print(dataset_file)
# # Global dict to save global results
time_vs_batch_size = {}
score_vs_batch_size = {}

# Load test dataset
mat = loadmat(dataset_file)
x_train = mat['x_train']
x_test = mat['x_test']
y_train = mat['y_train']
y_test = mat['y_test']
del mat
# normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape
# from shape (img_rows, img_cols, nb_of_images) to (nb_of_images, img_rows, img_cols)
x_train = np.rot90(np.rot90(x_train, axes=(2, 0)), axes=(1, 2))
# from shape (nb_of_image, img_rows, img_cols ) to (nb_of_image, img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = np.rot90(np.rot90(x_test, axes=(2, 0)), axes=(1, 2))
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# avoid float outputs
y_train[y_train != 1] = 0
y_test[y_test != 1] = 0

# Print out nice things
print("Dataset size")
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# reduction for test
y_train = y_train[:10]
x_train = x_train[:10]
y_test = y_test[:10]
x_test = x_test[:10]
# # Dataset ready

for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:

    compteur = 0
    score_liste = []
    time_liste = []
    print("batch_size = ", batch_size)

    # Creation of new model
    model = CNN_global(model_to_use)(input_shape, output_number)

    model.save(file_to_save_model + "_batch_size_" + str(batch_size) + "_.h5")
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print(model.metrics_names[0], " : ", score[0], " | ", model.metrics_names[1], " : ", score[1], )

    while score[1] <= .95 and compteur < 100:

        starting_time = time.time()
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)  # train model
        ending_time = time.time()

        score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
        print(model.metrics_names[0], " : ", score[0], " | ", model.metrics_names[1], " : ", score[1], end=" ")
        print("| time: ", float(ending_time - starting_time))

        score_liste.append(score)
        time_liste.append(float(ending_time - starting_time))
        compteur += 1

        # saving it to disk
        model.save(file_to_save_model + "_batch_size_" + str(batch_size) + "_.h5")

        # if no progress over 2 generations, STOP
        if stopTraning(score_liste):
            print("training is not efficient, lets stop there !")
            break  # while(score < )

    print("Model trained for %i epochs" % (compteur * 10))
    time_vs_batch_size[str(batch_size)] = time_liste
    score_vs_batch_size[str(batch_size)] = score_liste
    del model
    K.clear_session()
    print("###########################################")

save_obj(time_vs_batch_size, "time_vs_batch_size.txt")
save_obj(score_vs_batch_size, "score_vs_batch_size.txt")
