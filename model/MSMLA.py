"""
@author: Minho Kim
Contact : mhk93@snu.ac.kr

"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from utils import se_convolutional_block, se_identity_block
from utils import cbam_block as cbam


def MSMLA18(input_shape):

    # Input stage
    inputs = Input(input_shape)

    # Multi-scale layer
    x0 = Conv2D(16, (5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    x1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x2 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    
    # Fuse to multi-scale features (dim: 64)
    x3 = Concatenate(axis=-1)([x0, x1, x2])

    # Multi-Level Attention Layer (Branched Unit)
    in_cbam = cbam(x3, 'xCBAM')
    in_cbam = GlobalAveragePooling2D()(in_cbam)

    # SE-ResBlock 1 (16 filters) in main backbone & MLA (16 filters) in branch
    X = se_convolutional_block(x3, f=3, filters=[depth[0], depth[0] * 4], stage=2, block='a', s=1)
    cbam1 = cbam(X, 'cbam1')
    cbam1 = GlobalAveragePooling2D()(cbam1)
    X = se_identity_block(X, 3, [depth[0], depth[0] * 4], stage=2, block='b')

    # SE-ResBlock 2 (32 filters) in main backbone & MLA (32 filters) in branch
    X = se_convolutional_block(X, f=3, filters=[depth[1], depth[1] * 4], stage=3, block='a', s=2)
    cbam2 = cbam(X, 'cbam2')
    cbam2 = GlobalAveragePooling2D()(cbam2)
    X = se_identity_block(X, 3, [depth[1], depth[1] * 4], stage=3, block='b')

    # SE-ResBlock 3 (64 filters) in main backbone & MLA (64 filters) in branch
    X = se_convolutional_block(X, f=3, filters=[depth[2], depth[2] * 4], stage=4, block='a', s=2)
    cbam3 = cbam(X, 'cbam3')
    cbam3 = GlobalAveragePooling2D()(cbam3)
    X = se_identity_block(X, 3, [depth[2], depth[2] * 4], stage=4, block='b')

    # Context aggregation to create multi-level attention features (dim: 240)
    X = GlobalAveragePooling2D()(X)
    X = Concatenate(axis=-1)([X, in_cbam, cbam1, cbam2, cbam3])

    # FC layer for LCZ classification
    X = Dense(17, activation='softmax', name='fc' + str(17), kernel_initializer='he_normal')(X)

    # Create model
    model = Model(inputs=inputs, outputs=X, name='MSMLA-18')

    return model


def MSMLA50(input_shape):

    # Input stage
    inputs = Input(input_shape)

    # Multi-scale layer
    x0 = Conv2D(16, (5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    x1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x2 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # Fuse to multi-scale features (dim: 64)
    x3 = Concatenate(axis=-1)([x0, x1, x2]) 

    # Multi-Level Attention Layer (Branched Unit)
    in_cbam = cbam(x3, 'xCBAM')
    in_cbam = GlobalAveragePooling2D()(in_cbam)

    # SE-ResBlock 1 (16 filters) in main backbone & MLA (16 filters) in branch
    X = se_convolutional_block(x3, f=3, filters=[depth[0], depth[0], depth[0] * 4], stage=2, block='a', s=1)
    cbam1 = cbam(X, 'cbam1')
    cbam1 = GlobalAveragePooling2D()(cbam1)
    X = se_identity_block(X, 3, [depth[0], depth[0], depth[0] * 4], stage=2, block='b')
    X = se_identity_block(X, 3, [depth[0], depth[0], depth[0] * 4], stage=2, block='c')
  
    # SE-ResBlock 2 (32 filters) in main backbone & MLA (32 filters) in branch
    X = se_convolutional_block(X, f=3, filters=[depth[1], depth[1], depth[1] * 4], stage=3, block='a', s=2)
    cbam2 = cbam(X, 'cbam2')
    cbam2 = GlobalAveragePooling2D()(cbam2)
    X = se_identity_block(X, 3, [depth[1], depth[1], depth[1] * 4], stage=3, block='b')
    X = se_identity_block(X, 3, [depth[1], depth[1], depth[1] * 4], stage=3, block='c')
    X = se_identity_block(X, 3, [depth[1], depth[1], depth[1] * 4], stage=3, block='d')
   
    # SE-ResBlock 3 (64 filters) in main backbone & MLA (64 filters) in branch
    X = se_convolutional_block(X, f=3, filters=[depth[2], depth[2], depth[2] * 4], stage=4, block='a', s=2)
    cbam3 = cbam(X, 'cbam3')
    cbam3 = GlobalAveragePooling2D()(cbam3)
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='b')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='c')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='d')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='e')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='f')

    # Context aggregation to create multi-level attention features (dim: 240)
    X = GlobalAveragePooling2D()(X)
    X = Concatenate(axis=-1)([X, in_cbam, cbam1, cbam2, cbam3])

    # FC layer for LCZ classification
    X = Dense(17, activation='softmax', name='fc' + str(17), kernel_initializer='he_normal')(X)

    # Create model
    model = Model(inputs=inputs, outputs=X, name='MSMLA-50')

    return model


