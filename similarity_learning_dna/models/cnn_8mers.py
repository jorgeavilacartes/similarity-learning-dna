from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten, GlobalAveragePooling2D, Add, Activation, BatchNormalization, ZeroPadding2D

# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

# config
k = 8 # (256x256)

def identity_block(inp, filters, kernel_size, block, layer):
    
    f1, f2, f3 = filters
    
    conv_name = 'id_conv_b' + block + '_l' + layer
    batch_name = 'id_batch_b' + block + '_l' + layer
    
    x = Conv2D(filters=f1, kernel_size=1, padding='same', kernel_initializer='he_normal', name=conv_name + '_a')(inp)
    x = BatchNormalization(name=batch_name + '_a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name + '_b')(x)
    x = BatchNormalization(name=batch_name + '_b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=f3, kernel_size=1, padding='same', kernel_initializer='he_normal', name=conv_name + '_c')(x)
    x = BatchNormalization(name=batch_name + '_c')(x)
    
    add = Add()([inp, x])
    x = Activation('relu')(add)
    
    return x


def convolutional_block(inp, filters, kernel_size, block, layer, strides=2):
    
    f1, f2, f3 = filters
    
    conv_name = 'res_conv_b' + block + '_l' + layer
    batch_name = 'res_batch_b' + block + '_l' + layer
    
    y = Conv2D(filters=f1, kernel_size=1, padding='same', strides=strides, kernel_initializer='he_normal', name=conv_name + '_a')(inp)
    y = BatchNormalization(name=batch_name + '_a')(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters=f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name + '_b')(y)
    y = BatchNormalization(name=batch_name + '_b')(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters=f3, kernel_size=1, padding='same', kernel_initializer='he_normal', name=conv_name + '_c')(y)
    y = BatchNormalization(name=batch_name + '_c')(y)
    
    shortcut = Conv2D(filters=f3, kernel_size=1, strides=strides, kernel_initializer='he_normal', name=conv_name + '_shortcut')(inp)
    shortcut = BatchNormalization(name=batch_name + '_shortcut')(shortcut)
    
    add = Add()([shortcut, y])
    y = Activation('relu')(add)
    
    return y

def get_model():

    inp = Input(shape=(256, 256, 1), name='input')
    padd = ZeroPadding2D(3)(inp)

    conv1 = Conv2D(64, 7, strides=2, padding='valid', name='conv1')(padd)
    conv1 = BatchNormalization(name='batch2')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding2D(1)(conv1)
    conv1 = MaxPool2D(3, 2)(conv1)

    conv2 = convolutional_block(conv1, [64,64,256], 3, '2', '1', strides=1)
    conv2 = identity_block(conv2, [64,64,256], 3, '2', '2')
    conv2 = identity_block(conv2, [64,64,256], 3, '2', '3')

    conv3 = convolutional_block(conv2, [128,128,512], 3, '3', '1')
    conv3 = identity_block(conv3, [128,128,512], 3, '3', '2')
    conv3 = identity_block(conv3, [128,128,512], 3, '3', '3')
    conv3 = identity_block(conv3, [128,128,512], 3, '3', '4')

    conv4 = convolutional_block(conv3, [256,256,1024], 3, '4', '1')
    conv4 = identity_block(conv4, [256,256,1024], 3, '4', '2')
    conv4 = identity_block(conv4, [256,256,1024], 3, '4', '3')
    conv4 = identity_block(conv4, [256,256,1024], 3, '4', '4')
    conv4 = identity_block(conv4, [256,256,1024], 3, '4', '5')
    conv4 = identity_block(conv4, [256,256,1024], 3, '4', '6')

    conv5 = convolutional_block(conv4, [512,512,2048], 3, '5', '1')
    conv5 = identity_block(conv5, [512,512,2048], 3, '5', '2')
    conv5 = identity_block(conv5, [512,512,2048], 3, '5', '3')

    avg_pool = GlobalAveragePooling2D()(conv5)
    embeddings = Flatten()(avg_pool)
    embeddings = tf.keras.layers.Dense(256, activation=None)(avg_pool), # No activation on final dense layer
    embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings) # L2 normalize embeddings

    model = Model(inp, embeddings)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tfa.losses.TripletSemiHardLoss(),
                metrics=["accuracy"]
    )

    return model