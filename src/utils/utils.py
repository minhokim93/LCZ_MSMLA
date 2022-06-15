import numpy as np
import os
import rasterio

from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt

# 16bit to 8bit
def clip(img, percentile):
    out = np.zeros_like(img.shape[2])
    for i in range(img.shape[2]):
        a = 0 
        b = 255 
        c = np.percentile(img[:,:,i], percentile)
        d = np.percentile(img[:,:,i], 100 - percentile)        
        t = a + (img[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        img[:,:,i] =t
    rgb_out = img.astype(np.uint8)   
    return rgb_out

def min_max_normalization(img, percentile):
    img = img.astype(np.float32)
    out = np.zeros_like(img.shape[2])
    for i in range(img.shape[2]):
        a = 0 
        b = 1
        c = np.percentile(img[:,:,i], percentile)
        d = np.percentile(img[:,:,i], 100 - percentile)        
        t = a + (img[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        img[:,:,i] =t
        
    return img

def gee_normalization(img, mins=0.0, maxs=0.0):
    '''
    https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
    TOA reflectance scaled by 10000    
    '''
    img = np.copy(img).astype(np.float32)
    # out = np.zeros_like(img.shape[2])
    for i in range(img.shape[2]):
        t = img[:,:,i]
        
        # t = a + (img[:,:,i] - c) * (b - a) / (d - c) 
        t[t<1] = 1 # minimum value of TOA reflectance = 1
        t[t>10000] = 10000 # maximum value of TOA reflectance = 10000
        img[:,:,i] = t
        
    img = img/10000
    return img


def patches(crop_img, crop=32):
    patch = []
    for i in range(int(len(crop_img)/crop)):
        for j in range(int(len(crop_img)/crop)):
            # cropped.append(crop_img[x:(i+1)*crop, y:(i+1)*crop]) #"i+1" used to start from index 1
            cropped_img = (crop_img[i*crop:(i+1)*crop, j*crop:(j+1)*crop]) #"i+1" used to start from index 1
            # patch_img = np.rollaxis(cropped_img,2,0) #  transpose
            # crop_list = cropped_img.tolist()
            patch.append(cropped_img)
            # path = direc + 's2_'+img_name.split('_')[1] + '_crop_'+ str(i) + '.tif'
            # imsave(path, patch_img)
    patch = np.asarray(patch)

    return patch 

def patch_overlap(img,crop=32, overlap=22):
    
    shape = img.shape
    delta = crop - overlap
    
    quotient_y = int(shape[0])//delta - 1
    quotient_x = int(shape[1])//delta - 1
    
    yy = np.linspace(0,quotient_y,quotient_y+1) * delta
    xx = np.linspace(0,quotient_x,quotient_x+1) * delta
    
    while int(shape[0])-yy[-1]<crop:
        yy = np.delete(yy, -1)

    while int(shape[1])-xx[-1]<crop:
        xx = np.delete(xx, -1)
        
    yy = yy.astype('uint16')
    xx = xx.astype('uint16')
    
    patch_stack = []
    for ii in yy:
        for jj in xx:
            cropped_img = img[ii:ii+crop,jj:jj+crop,:]
            patch_stack.append(cropped_img)
            
    patch_stack = np.asarray(patch_stack)
    info_shape = np.array([len(yy), len(xx)])
    
    return patch_stack, info_shape # np.


###################################
## for train $ test
###################################
def load_cropped(vnir_path, percentile=0.0, num_classes=17):
    snir_path = vnir_path.replace('vnir', 'swir')
    acquisition_date = os.path.splitext(os.path.split(vnir_path)[1])[0].split('_')[1]

    with rasterio.open(vnir_path, 'r') as src:
        vnir = src.read().transpose(1,2,0) # [col, row, band] # RGBN
        meta = src.meta
    with rasterio.open(snir_path, 'r') as src:
        swir = src.read().transpose(1,2,0) # [col, row, band]
    
    img = vnir[...,:3]
    img = np.concatenate([img, swir[...,:3]], axis=-1)
    img = np.concatenate([img, vnir[...,3:]], axis=-1)
    img = np.concatenate([img, swir[...,3:]], axis=-1)
    
    # img = np.concatenate((vnir, snir), axis = 2) # [vnir, snir]
    img = gee_normalization(img, mins=0.0, maxs=percentile) # [0,1] normalization, percentile: variable
    img = np.pad(img, ((11,11),(11,11),(0,0)), mode='reflect')

    # 2. slice [:, 32,32,bands]
    x_tst, info_shape = patch_overlap(img, crop=32, overlap=22) # [batch_indices,32,32,bands]
           
    return x_tst, acquisition_date

def prepare_label(gt_path='../Ground Truth/gt_seoul_ready_subset.tif', num_classes=17):
    with rasterio.open(gt_path, 'r') as src:
        gt_raster = src.read(1)
        
    y_flatten = gt_raster.flatten()
    if num_classes==17:
        y_tst = np.zeros(shape=(len(y_flatten), num_classes))
        for i, value in enumerate(y_flatten):
            y_tst[i, value-1] = 1
            
    else:
        #  num_classes = len(location)
        y_tst = np.zeros(shape=(len(y_flatten), len(target_class)))
        for i, value in enumerate(y_flatten):
            location = target_class.index(value)
            print('unique number: {}'.format(location))
            y_tst[i, location] = 1
    return y_tst.astype('int64')


def gt_proc(img_gt):
    '''
    LCZ_A=101 로 저장되어있는 것을 LCZ_A=11로 변환
    'float32' -> 'int16'
    '''
    # Update LCZ class values(For natural land cover classes)
    # img_gt[img_gt==-1] = 0 # No data
    img_gt[img_gt==101] = 11
    img_gt[img_gt==102] = 12
    img_gt[img_gt==103] = 13
    img_gt[img_gt==104] = 14
    img_gt[img_gt==105] = 15
    img_gt[img_gt==106] = 16
    img_gt[img_gt==107] = 17
    values, counts = np.unique(img_gt, return_counts=True)
    print('Number of valid classes :', values.shape[0])
    print('Unique LCZ classes are :', values)
    print('1=LCZ1, 2=LCZ2, 3=LCZ3, 4=LCZ4, 5=LCZ5, 6=LCZ6, 7=LCZ7, 8=LCZ8, 9=LCZ9, 10=LCZ10')
    print('11=LCZ_A, 12=LCZ_B, 13=LCZ_C, 14=LCZ_D, 15=LCZ_E, 16=LCZ_F, 17=LCZ_G')
    
    return img_gt, values

def plot_confusion_matrix(cfm,name,dpi=300):
    lcz = ['LCZ-1','LCZ-2','LCZ-3','LCZ-4','LCZ-5','LCZ-6','LCZ-7','LCZ-8','LCZ-9',
           'LCZ-10','LCZ-A','LCZ-B','LCZ-C','LCZ-D','LCZ-E','LCZ-F','LCZ-G']
    plt.figure(figsize=(9,9),dpi=dpi)
    
    import pandas as pd
    conf = pd.DataFrame(cfm)
    
    zero_cols = [ col for col, is_zero in ((conf == 0).sum() == conf.shape[0]).items() if is_zero ]
    
    conf.drop(zero_cols, axis=0, inplace=True)
    conf.drop(zero_cols, axis=1, inplace=True)
    print("Number of classified LCZs:",len(conf), "and classes removed :", zero_cols)
    
    if len(zero_cols) > 1:
        for idx, val in enumerate(zero_cols):
            lcz.remove(lcz[val])
            print(lcz)
    elif len(zero_cols) == 1:
        lcz.remove(lcz[int(zero_cols[len(zero_cols)-1])])
        print(lcz)
    
    cfm = np.asarray(conf)
        
    imx = conf.shape[0]
    cm = np.zeros([imx,imx])
    for i in range(imx):
        cm[:,i] = cfm[:,i]/cfm[:,i].sum()*100
    plt.imshow(cm,interpolation='nearest')
    plt.title(name)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks=np.arange(imx)
    plt.xticks(tick_marks,np.arange(imx)+1,fontsize=10,rotation=45)
    plt.yticks(tick_marks,np.arange(imx)+1,fontsize=10,rotation=45)
    plt.ylabel('Predicted Label')
    plt.xlabel('Reference Label')
    fmt = '.0f'
    thresh = cm.max() / 2.
    for i in range(imx):
        for j in range(imx):
            plt.text(j, i, format(cfm[i,j], fmt),
                    ha="center", va="center",fontsize=12,
                    color="white" if cm[i,j] < thresh else "black")
    plt.xlim(-0.5,imx-0.5)
    plt.ylim(imx-0.5,-0.5)
    
    plt.xticks(np.arange(0,len(cfm)),lcz)
    plt.yticks(np.arange(0,len(cfm)),lcz)
    plt.tight_layout()
    # plt.savefig(name+'.png', dpi=dpi)
    # plt.savefig(name, dpi=300, bbox_inches='tight')

    plt.show()
    return cfm



def lcz_png_maker(lcz_path, lcz_map, model_name, city):
    values, counts = np.unique(lcz_map, return_counts=True)
### Plotting LCZ ground truth

    colors = []
    actual_labels = []
    labels = list(values.astype('int'))

        # Dictionary of LCZ names, numerical labels, and color-code
    lcz_dict = {
            "LCZ1": {
                "name":'LCZ 1',
                "label":1,
                "color":'#8c0000'
                },
            "LCZ2": {
                "name":'LCZ 2',
                "label":2,
                "color":'#d10000'
                },
            "LCZ3": {
                "name":'LCZ 3',
                "label":3,
                "color":'#ff0000'
                },
            "LCZ4": {
                "name":'LCZ 4',
                "label":4,
                "color":'#bf4d00'
                },
            "LCZ5": {
                "name":'LCZ 5',
                "label":5,
                "color":'#ff6600'
                },
            "LCZ6": {
                "name":'LCZ 6',
                "label":6,
                "color":'#ff9955'
                },
            "LCZ7": {
                "name":'LCZ 7',
                "label":7,
                "color":'#faee05'
                },
            "LCZ8": {
                "name":'LCZ 8',
                "label":8,
                "color":'#bcbcbc'
                },
            "LCZ9": {
                "name":'LCZ 9',
                "label":9,
                "color":'#ffccaa'
                },
            "LCZ10": {
                "name":'LCZ 10',
                "label":10,
                "color":'#555555'
                },
            "LCZ11": {
                "name":'LCZ A',
                "label":11,
                "color":'#006a00'
                },
            "LCZ12": {
                "name":'LCZ B',
                "label":12,
                "color":'#00aa00'
                },
            "LCZ13": {
                "name":'LCZ C',
                "label":13,
                "color":'#648525'
                },
            "LCZ14": {
                "name":'LCZ D',
                "label":14,
                "color":'#b9db79'
                },
            "LCZ15": {
                "name":'LCZ E',
                "label":15,
                "color":'#000000'
                },
            "LCZ16": {
                "name":'LCZ F',
                "label":16,
                "color":'#fbf7ae'
                },
            "LCZ17": {
                "name":'LCZ G',
                "label":17,
                "color":'#6a6aff'
                }
            }
    
    # Find matching labels
    for i in labels:
        for lcz,v in lcz_dict.items():
            if i == v["label"]:
                actual_labels.append(lcz_dict['LCZ'+str(i)]['name'])
                colors.append(lcz_dict['LCZ'+str(i)]['color'])


    from matplotlib_scalebar.scalebar import ScaleBar

    plt.ion() # interactive-on 대화형 모드 켬, plt.show() 없이 자동으로 출력 (<->plt.ioff: plt.show()있는경우만 출력)
    plt.clf() # clear figure: 전체 그림 삭제
    fig=plt.figure(1)
    cmap = ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(labels, cmap.N)
    plot2=plt.imshow(lcz_map,cmap, norm)
    plt.xticks([])
    plt.yticks([])
    
    scalebar = ScaleBar(10) # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    

    # # cbar=plt.colorbar(plot2, orientation = 'horizontal', fraction=0.046, pad=0.04)
    # tick_locs = (np.arange(len(actual_labels)) + 1.5)
    # cbar.set_ticks(tick_locs)
    # cbar.ax.set_yticklabels(actual_labels, fontsize=15, ha = 'left')   
    # cbar.ax.axes.tick_params(length=0)


    plt.draw() # plot 그리기. plt.show()와 달리 화면에 출력되지 않음
    # plt.title("Ground Truth of "+city+" (100m)")
    plt.savefig(lcz_path, dpi=300, bbox_inches='tight')


import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = Precision(thresholds=0.5)
        self.recall_fn = Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
