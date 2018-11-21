from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose
from keras.layers import Concatenate, Add, Average,Convolution2D
from keras import backend as K

from keras.utils import plot_model
from keras.models import load_model

import sys

modelArch = {}												
addModel = lambda f:modelArch.setdefault(f.__name__,f)

# build and return model
def makeModel(architecture, params=None, verbose=False):

    model = modelArch[architecture](params)
    if verbose:
        print(model.summary(line_length=150))
        plot_model(model, to_file='./model_%s.png'%architecture)

    return model

@addModel
def ResNetSR001(params):

    def residual_block(x, nFilters, label):

        ''' this block performs 3x3 convolution twice on the input and adds the result to the input '''

        x0 = x
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(nFilters, (3, 3), activation='linear', padding='same', name='sr_res_conv_' + label + '_1')(x)
        x = BatchNormalization(axis=channel_dim, name="sr_res_batchnorm_" + label + "_1")(x)
        x = Activation('relu', name="sr_res_activation_" + label + "_1")(x)

        x = Conv2D(nFilters, (3, 3), activation='linear', padding='same', name='sr_res_conv_' + label + '_2')(x)
        x = BatchNormalization(axis=channel_dim, name="sr_res_batchnorm_" + label + "_2")(x)

        m = Add(name="sr_res_merge_" + label)([x, x0])

        return m

    def upscale_block(x, label, useConv2DTranspose=False):
        
        ''' this block upscales the input tensor x by a factor of 2. If useConv2DTranspose set to True,
        tranpose convolution is used for the upsampling. Otherwise, a upsampling by insterting zeros
        followed by normal convolution is used. The number of output filters is the same as the number of 
        input channels/filters '''

        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = x._keras_shape[channel_dim]

        if useConv2DTranspose:
            x = Conv2DTranspose(channels, (4, 4), strides=(2, 2), padding='same', activation='relu',
                                        name='upsampling_deconv_%d' % label)(x)
        else:
            x = UpSampling2D()(x)
            x = Conv2D(channels, (3, 3), activation="relu", padding='same', name='sr_res_filter1_' + label)(x)
        
        return x

    i = Input(shape=(32,32,3), name="Input")

    x0 = Conv2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(i)

    x = residual_block(x0, nFilters=64, label='1')

    nb_residual = 5
    for idx in range(nb_residual):
        x = residual_block(x, nFilters=64, label='%d' % (idx+2) )

    x = Add()([x, x0])
    x = upscale_block(x, label='1', useConv2DTranspose=False)
    x = Conv2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

    return Model(inputs=[i], outputs=[x])

@addModel
def SRCNN(n1=64,n2=32,n3=1,f1=9,f2=1,f3=5,img_rows=32,img_cols=32,channels=3):
    x = Input(shape = (img_rows,img_cols,channels))
    c1 = Convolution2D(n1, f1,f1, activation = 'relu', init = 'he_normal', border_mode='same')(x)
    c2 = Convolution2D(n2, f2, f2, activation = 'relu', init = 'he_normal', border_mode='same')(c1)
    c3 = Convolution2D(n3, f3, f3, init = 'he_normal', border_mode='same')(c2)
    model = Model(input = x, output = c3)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    #model.compile(loss='mse', metrics=[PSNRLoss], optimizer=adam)     
    return model

if __name__ == '__main__':
	model = makeModel(sys.argv[1], verbose=True)
	plot_model(model, to_file='./model_%s.png' % sys.argv[1])
	print("\nLayers = %d" % len(model.layers))
