import keras
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, SeparableConvolution2D, \
    Activation, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD


def CNN_global(numero):
    switcher = {
        1: CNN_VGG,
        2: CNN_Wolkow,
        3: DeepDog,
        'Deepdog': DeepDog, 'DeepDog': DeepDog, 'deepdog': DeepDog,
        'CNN_VGG': CNN_VGG,
        'CNN_Wolkow': CNN_Wolkow, 'CNN_wolkow': CNN_Wolkow,
        'DeepDogSource': DeepDogSource,
    }
    # Get the function from switcher dictionary
    func = switcher.get(numero, lambda: "Invalid month")
    # return the function
    return func


def CNN_VGG(input_shape, output_number):
    """
    Returns a keras CNN model compiled, similar to the one
    described on the Keras website (Guide to the Sequential mode / Example / VGG like convnet
    https://keras.io/getting-started/sequential-model-guide/#examples
    Consulted in June 2018
    """
    # Creation of new model
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_number, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


def CNN_Wolkow(input_shape, output_number):
    """
    Returns a keras CNN model compiled, similar to the one
    described in the Wolkow and al. publication :
    "Autonomous Scanning Probe Microscopy in Situ Tip Conditioning through Machine Learning"
    Published in 03/2018 in ACSNano
    """
    # Creation of new model
    model = Sequential()
    # input: 32x32 images with 3 channels -> (32, 32, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(30, (5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(40, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_number, activation='softmax'))

    adam_optimizer = keras.optimizers.Adam(lr=10 ** -4)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam_optimizer,
                  metrics=['accuracy'])

    return model


def DeepDog(input_shape, output_number):
    """
    Returns a keras CNN model compiled, similar to the one
    described on the Deepdog project website :
    https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3
    Consulted in June 2018
    """
    model = Sequential(name='deepdog')

    model.add(Convolution2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    for _ in range(5):
        model.add(SeparableConvolution2D(512, (3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))

    model.add(SeparableConvolution2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConvolution2D(1024, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(output_number, activation='sigmoid'))  # output layer

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def DeepDogSource(input_shape, output_number):
    """
    Returns a keras CNN model compiled, similar to the one
    described on the Deepdog project website :
    https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3
    Consulted in June 2018
    """
    input_tensor = None
    #     input_shape=input_shape
    alpha = 1
    classes = 1000
    #     input_shape = _obtain_input_shape(input_shape,
    #                                       default_size=224,
    #                                       min_size=32,
    #                                       data_format=K.image_data_format(),
    #                                       require_flatten=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    for _ in range(5):
        x = SeparableConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)

    x = SeparableConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConvolution2D(int(1024 * alpha), (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)
    out = Dense(output_number, activation='sigmoid')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='deepdog')

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
