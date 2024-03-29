"""
@author: Minho Kim
Contact : mhk93@snu.ac.kr

"""
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

# Call models

def get_model(model, input_shape, d):
        
    # RESNET MODELS
    if model == 'MSMLA18':
        from model.MSMLA import MSMLA18
        net = MSMLA18(input_shape, [d, 2*d, 3*d]) # Depth [d, 2*d, 3*d] inputted into utils se_convolutional_block and se_identity_block
        net.build(input_shape)
    elif model == 'MSMLA50':
        from model.MSMLA import MSMLA50
        net = MSMLA50(input_shape, [d, 2*d, 3*d])
        net.build(input_shape)

    inputs = Input(shape=input_shape)
    outputs = net(inputs)
    
    return Model(inputs=inputs, outputs=outputs)