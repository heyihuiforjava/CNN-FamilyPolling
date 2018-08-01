from keras.utils import plot_model

from CNN_generation import *

# This modul needs the pydot module and the Graphviz executable in the PATH

if __name__ == "__main__":
    img_rows = img_cols = image_size = 128
    output_number = 1
    input_shape = (img_rows, img_cols, 1)  # image image_size*image_size greyscale
    model2use = 'Deepdog'
    model = CNN_global(model2use)(input_shape, output_number)

    plot_model(model, to_file=model2use + '.png')
