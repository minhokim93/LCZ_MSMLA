import tensorflow as tf
from tensorflow.keras.layers import *


### 1. ResBlock

def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X

    if len(filters) > 2:

        X = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    else:

        X = Conv2D(filters=filters[0], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[0], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X

    if len(filters) > 2:
        X = Conv2D(filters[0], (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X_shortcut = Conv2D(filters=filters[-1], kernel_size=(1, 1), strides=(s, s), padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer='he_normal')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    else:

        X = Conv2D(filters[0], (3, 3), strides=(s, s), padding='same', name=conv_name_base + '2a',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[0], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

        X_shortcut = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(s, s), padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer='he_normal')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

### 2. SE-ResBlock

def se_block(block_input, ratio=8):
    filter_kernels = block_input.shape[-1]
    z_shape = (1, 1, filter_kernels)
    z = GlobalAveragePooling2D()(block_input)
    z = Reshape(z_shape)(z)
    s = Dense(filter_kernels // ratio, activation='relu')(z)
    s = Dense(filter_kernels, activation='sigmoid')(s)
    x = multiply([block_input, s])

    return x


def se_identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X

    if len(filters) > 2:

        X = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    else:

        X = Conv2D(filters=filters[0], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[0], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    se = se_block(X)

    X = Add()([se, X_shortcut])
    X = Activation('relu')(X)

    return X


def se_convolutional_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X

    if len(filters) > 2:

        X = Conv2D(filters[0], (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X_shortcut = Conv2D(filters=filters[-1], kernel_size=(1, 1), strides=(s, s), padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer='he_normal')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    else:

        X = Conv2D(filters[0], (1, 1), strides=(s, s), padding='same', name=conv_name_base + '2a',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters[0], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

        X_shortcut = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(s, s), padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer='he_normal')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    se = se_block(X)

    X = Add()([se, X_shortcut])
    X = Activation('relu')(X)

    return X

### 3. CBAM

def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.compat.v1.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')

    return attention_feature


def channel_attention(input_feature, name, ratio=4):

    kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None)

    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.compat.v1.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        # Average Pool
        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = Dense(units=channel // ratio, activation='relu', kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)(avg_pool)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = Dense(units=channel, activation='relu', kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)(avg_pool)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        # Max Pool
        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        max_pool = Dense(units=channel // ratio, activation='relu')(avg_pool)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)

        max_pool = Dense(units=channel, activation='relu')(avg_pool)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name, kernel_size=7):
    kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None)

    with tf.compat.v1.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = Conv2D(filters=1, kernel_size=[kernel_size, kernel_size], use_bias=False,
                        padding='same', activation=None, kernel_initializer=kernel_initializer
                        )(concat)
        assert concat.get_shape()[-1] == 1

        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat