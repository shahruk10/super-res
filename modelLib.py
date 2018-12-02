from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Activation, Subtract, add
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose
from keras.layers import Concatenate, Add, Average,Convolution2D
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.utils import plot_model
from keras.models import load_model
from customInit import bilinear_upsample_weights
from keras.initializers import Constant


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
    x = Conv2D(3, (3, 3), activation="sigmoid", padding='same', name='sr_res_conv_final')(x)

    return Model(inputs=[i], outputs=[x])

@addModel
def SRCNN_001(n1=64,n2=32,n3=1,f1=9,f2=1,f3=5,img_rows=32,img_cols=32,channels=3):
    x = Input(shape = (img_rows,img_cols,channels))
    c1 = Convolution2D(n1, f1,f1, activation = 'relu', init = 'he_normal', border_mode='same')(x)
    c2 = Convolution2D(n2, f2, f2, activation = 'relu', init = 'he_normal', border_mode='same')(c1)
    c3 = Convolution2D(n3, f3, f3, init = 'he_normal', border_mode='same')(c2)
    model = Model(input = x, output = c3)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
    #model.compile(loss='mse', metrics=[PSNRLoss], optimizer=adam)     
    return model

@addModel
def DnCNN_001():
    inpt = Input(shape=(32,32,3))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    return model

@addModel
def VDSR_001(IMG_SIZE = (32,32,1)):
    input_img = Input(shape=IMG_SIZE)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model

    output_img = add([res_img, input_img])

    model = Model(input_img, output_img)

    return model

@addModel
def FSRCNN_001(scale_factor=4):
    input_img = Input(shape=(32,32, 1))

    model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = PReLU()(model)

    model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)

    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)
    model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)

    model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = PReLU()(model)

    model = Conv2DTranspose(1, (9, 9), strides=(scale_factor, scale_factor), padding='same')(model)

    output_img = model

    model = Model(input_img, output_img)
    return model

@addModel
def LapSRN_001(d=10,s=3):
    '''
    d = no of Conv Layers in each stage
    s = no of stages
    * custom initializer used : bilinear_upsample_weights
    '''
    def stage(x,xint,d):
        for i in range(d):
                x = Conv2D(64,(3,3), padding = 'same', kernel_initializer='he_normal')(x)
                x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(1,(4,4),strides = (2,2), padding = 'same', kernel_initializer = Constant(bilinear_upsample_weights(2,1)))(x)
        res = Conv2D(1,(3,3), padding = 'same', kernel_initializer='he_normal')(x)
        up = Conv2DTranspose(1,(4,4),strides = (2,2), padding = 'same', kernel_initializer = Constant(bilinear_upsample_weights(2,1)))(xint)
        out = add([up,res])
        return x, out

    xin = Input(shape =(32,32,1))
    x = xin
    out = xin
    for i in range(s):
        x, out = stage(x, out, d)
    model = Model(inputs = xin, outputs = out)
    return model

if __name__ == '__main__':
	model = makeModel(sys.argv[1], verbose=True)
	plot_model(model, to_file='./model_%s.png' % sys.argv[1])
	print("\nLayers = %d" % len(model.layers))
