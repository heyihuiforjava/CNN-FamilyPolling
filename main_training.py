# importing the Networks related lib
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
from os import sep

from CNN_generation import CNN_global


def prepare_Dataset_bin(category):
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

    y_train = np.array([[1] if element[category] == 1 else [0] for element in y_train])
    y_test = np.array([[1] if element[category] == 1 else [0] for element in y_test])

    # Print out nice things
    print("Dataset size")
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    ##PROBLEM : ImageDataGenerator usage misunderstood, need to be changed
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=False,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    # datagen.fit(x_train)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # variables
    model_to_use = "Deepdog"  # must be one of the name in CNN_global() dict
    folder_to_save_model = "models" + os.sep
    file_to_save_model = "my_%s_model" % model_to_use
    nb_training_session = 1
    epoch_by_training_session = 1
    img_rows = img_cols = image_size = 128
    output_number = 1
    input_shape = (img_rows, img_cols, 1)  # image image_size*image_size greyscale

    # load the dataset from GDrive to VM
    dataset_file = 'datasets' + os.sep + 'Training_Images_%d_mean.mat' % image_size
    print(dataset_file)

    # #Training the family's members
    for data_category in [0, 1, 2, 3, 4, 5]:
        #  recover and format dataset for this data category
        (x_train, y_train), (x_test, y_test) = prepare_Dataset_bin(data_category)

        # Train 3 CNN for each category
        for modelNo in range(3):
            # Creation of new model
            model = CNN_global(model_to_use)(input_shape=input_shape, output_number=output_number)
            print(model)
            model_file_title = file_to_save_model + "_" + str(modelNo) + "_cat_" + str(data_category) + ".h5"
            model.save(folder_to_save_model + model_file_title)

            for _ in range(nb_training_session):
                model.fit(x_train, y_train,
                          batch_size=32,
                          epochs=epoch_by_training_session,
                          verbose=0, )
                score = model.evaluate(x_test, y_test, batch_size=512)
                print("score :", score)
                model.save(folder_to_save_model + model_file_title)
            # # Clear memory to avoid overflow
            del model
            K.clear_session()
            print("######################################################\n\n")

    print("End of training")
    # # End of Training
