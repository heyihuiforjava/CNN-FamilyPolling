import numpy as np
from PIL import Image
from scipy.io import loadmat


def saveImg(data,title="image.png",show=False):
    try:
        im = Image.frombytes(mode="L", data=data, size=(image_size, image_size))
    except:
        data = data.reshape(image_size,image_size)
        im = Image.fromarray(data, mode="L")
    im.save(title, "PNG")
    if show : im.show()


if __name__ == '__main__':
    image_size = img_rows = img_cols = 128

    dataset_file = '..\\datasets\\Training_Images_128_mean.mat'

    mat = loadmat(dataset_file)
    # x_train = mat['x_train']
    # y_train = mat['y_train']
    x_test = mat['x_test']
    y_test = mat['y_test']
    del mat  # free memory

    # x_train = np.rot90(np.rot90(x_train, axes=(2, 0)), axes=(1, 2))
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = np.rot90(np.rot90(x_test, axes=(2, 0)), axes=(1, 2))
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)


    for i in range(len(x_test)):
        saveImg(x_test[i],"titleSTM%d.png"%i)
        # print(y_test[i])
        # input()
