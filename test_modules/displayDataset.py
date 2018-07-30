import numpy as np
from PIL import Image
from scipy.io import loadmat


def dispImg(data):
    im = Image.frombytes(mode="L", data=data, size=(image_size, image_size))
    im.show()
    # im.save('resized_image.jpg', "JPEG")


if __name__ == '__main__':
    image_size = 128

    dataset_file = '..\\datasets\\Training_Images_128_mean.mat'

    mat = loadmat(dataset_file)
    x_train = mat['x_train']
    del mat  # free memory

    # recovering the image at constant z
    tmp = []
    for i in range(len(x_train)):
        tmp.append([])
        for j in range(len(x_train[0])):
            tmp[i].append(x_train[i][j][40])
    tmp = np.array(tmp)
    print(tmp.shape)
    print(tmp)
    dispImg(tmp)

    # bad Reshape # from Trainer_mean4.py
    x_train = x_train.reshape(x_train.shape[2], image_size, image_size, 1)

    print("after reshape")
    print("x_train[0].shape", x_train[0].shape)
    dispImg(x_train[3])

    print("x_train[0] == tmp ")
    print(x_train[0] == tmp)
