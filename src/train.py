"""
@author: Minho Kim
Contact : mhk93@snu.ac.kr

"""

### Libraries
import numpy as np
import matplotlib.pyplot as plt
import os, datetime, warnings, glob, json, h5py

### Deep learning libraries
import tensorflow # TF 2.1
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
# from tensorflow.keras.callbacks import CSVLogger ## For csv logging
import albumentations as A
from sklearn.utils import class_weight

### Custom libraries
from dataLoader import MyGenerator
import lr
from call_model import get_model

import tensorflow_addons as tfa # Use from metric.metric import F1_Score if tensorflow_addons does not work

# Random seeds for reproducibility
SEED = 123
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
print('Using device:', gpus)

# Main Functions
def checknum(filename):
    test_filename = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('train', 'val'))

    with h5py.File(filename,'r') as f:
        bx = f['label']
        class_weights = class_weight.compute_sample_weight('balanced', bx)
        trainNumber = len(list(f['label']))
    with h5py.File(test_filename, "r") as ft:
        validationNumber = len(list(ft['label']))
    
    return trainNumber,validationNumber,class_weights
  
        
def jsondump(model, jsonfile, filename):
    
    test_filename = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('train', 'val'))
    
    with open(jsonfile) as f:
        config = json.load(f)
    
    ### Args to be modified in json file 
    config['name'] = model
    config['arch']['type'] = model
    config['data_loader']['args']['data_dir'] = filename

    with h5py.File(filename,'r') as f:
        init_band = int(f['sen2'].shape[3])
        window = int(f['sen2'].shape[2])
        
    config['val_data_loader']['args']['data_dir'] = test_filename
    config['arch']['args'] = {'init_band':init_band, 'window':window}    

    with open(jsonfile, 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=4) 
        
        
def train(model,trainNumber,validationNumber,class_weights,filename,batchsize,filter_depth):
    train_file = filename
    validation_file = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('train', 'val'))

    with open(jsonfile) as f:
        config = json.load(f)

    input_shape=config['arch']['args']['window'], config['arch']['args']['window'],config['arch']['args']['init_band'] 
     
    # Define model
    net = get_model(model=config['name'], input_shape=input_shape,d=filter_depth)
    net.compile(optimizer = tensorflow.keras.optimizers.Nadam(), loss = 'categorical_crossentropy', 
                metrics=['accuracy','Precision','Recall', # F1 score reports only positive classes
                         tfa.metrics.F1Score(num_classes=17,average='weighted',name='f1_weighted')]) 
                         # Weighted F1 reports score of positive and negative classes and is important for imbalanced classes


    # Augmentation
    augmentations = A.Compose([A.HorizontalFlip(p=0.5),A.Rotate(limit=(-90, 90)),A.VerticalFlip(p=0.5)])  
    
    # Hyperparameters
    lr_sched = lr.step_decay_schedule(initial_lr=0.001, decay_factor=0.004, step_size=5)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 15, verbose=1)    
    tb_name = model+'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs=100  
    
    # Directories
    # csv_logger = CSVLogger("saved/log/"+tb_name+".csv", append=True)
    modelbest = 'saved/model/'+tb_name+'.hdf5'
    checkpoint = ModelCheckpoint(filepath=modelbest, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    # Train    
    history = net.fit_generator(MyGenerator(train_file, batch_size=batchsize, augmentations=augmentations, shuffle=True),
                        workers=1,
                        steps_per_epoch = trainNumber//batchsize,                    
                        validation_data = MyGenerator(validation_file, batch_size=batchsize, augmentations=augmentations, shuffle=True),
                        validation_steps = validationNumber//batchsize,
                        epochs=epochs,
                        max_queue_size=100,
                        class_weight = class_weights,
                        callbacks=[early_stopping, lr_sched,checkpoint]) # If using csv logging, add csv_logger

    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.savefig('saved/log/'+tb_name+'.png', dpi = 300)
    plt.show()

    # net.save(modelbest)


### Main Processing
os.chdir(r'E:/ISPRS') # Change to base directory
jsonfile = r'E:/ISPRS/config_massive.json' # Set json path to call parameters (config_massive.json)
dataset_path = r'E:\ISPRS\geographic_split/onlyaux' # Path to .h5 files

# Load filenames
filelist = glob.glob(dataset_path + '/*.h5')
filelist = [x for x in filelist if 'train' in x]
train_filelist = [x for x in filelist if '_all_win' in os.path.basename(x)]

print('\n'.join(train_filelist))

# Train models
for model in ['MSMLA18', 'MSMLA50']:

    for filename in train_filelist:
        print(filename)
        jsondump(model, jsonfile, filename) 
        trainNumber,validationNumber,class_weights = checknum(filename)
        train(model,trainNumber,validationNumber,class_weights,filename,batchsize=32,filter_depth=16)
   
