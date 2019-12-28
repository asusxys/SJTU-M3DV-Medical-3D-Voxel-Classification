'''
加载已经训练好的模型进行预测
'''

import os
import numpy as np
import pandas as pd
from dataloader import *
from keras.models import load_model
from mylib.models import densesharp, metrics, losses
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

path = "./dataset/"
TEST = pd.read_csv(os.path.join(path, 'test.csv'))

def get_oneModel(n, crop_size, learning_rate, segmentation_task_ratio, weight_decay):

    test_dataset = ClfSegTestDataset(crop_size=crop_size,move=None)
    test_loader = get_loader_inorder(test_dataset, batch_size=1)
    model = densesharp.get_compiled(output_size=1,
                                optimizer=Adam(lr=learning_rate),
                                loss={"clf": 'binary_crossentropy',
                                      "seg": losses.DiceLoss()},
                                metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure, metrics.auc],
                                         'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                loss_weights={"clf": 1., "seg": segmentation_task_ratio},
                                weight_decay=weight_decay)

    if n==0:
        model.load_weights('./tmp/test/weights24_222593.h5')
    else:
        model.load_weights('./tmp/test/weights42_222639.h5')
    
    pred = model.predict_generator(generator=test_loader,steps=len(test_dataset), verbose=1)
    #print(pred[0])
    
    predict = pred[0][:,0]
    return predict
 
def main(crop_size, learning_rate, segmentation_task_ratio, weight_decay):
    res1 = get_oneModel(0, crop_size, learning_rate, segmentation_task_ratio, weight_decay)
    res2 = get_oneModel(1, crop_size, learning_rate, segmentation_task_ratio, weight_decay)
    
    index = tuple(TEST.index)
    name = TEST.loc[index, 'name']
    name.tolist()

    avg_pred = (res1+res2)/2.0
    predict = avg_pred.tolist()
    data={'Id':name,'Predicted':predict}
    dt = pd.DataFrame(data = data,columns=['Id','Predicted'])
    dt.to_csv('Submission.csv',index=False)
     
 

if __name__ == '__main__':
   main(crop_size=[32, 32, 32],
        learning_rate=1.e-5,
        segmentation_task_ratio=0.2,
        weight_decay=0.0)
