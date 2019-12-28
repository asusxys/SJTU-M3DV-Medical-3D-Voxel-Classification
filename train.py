'''
Main Function File
- modified base on original version (author: @duducheng)
- this file defines the training process
'''
import os
import pandas as pd
import numpy as np

from dataloader import *
from keras.optimizers import Adam, SGD
from mylib.models.misc import set_gpu_usage

set_gpu_usage()    # use gpu to run the code

from mylib.models import densesharp, metrics, losses
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau,LearningRateScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

path = "./dataset/"
TEST = pd.read_csv(os.path.join(path, 'test.csv'))


def main(batch_size, crop_size, random_move,learning_rate,segmentation_task_ratio, weight_decay, save_folder, epochs, alpha):
    '''
    :param batch_size
    :param crop_size: the input size
    :param learning_rate: learning rate of the optimizer
    :param segmentation_task_ratio: the weight of segmentation loss in total loss
    :param weight_decay: l2 weight decay
    :param save_folder: where to save the snapshots, tensorflow logs, etc.
    :param epochs: how many epochs to run
    :return:
    '''

    print(learning_rate)
    print(alpha)
    print(weight_decay)
    
    train_dataset = ClfSegDataset(subset=[3, 4, 0, 1])
    train_loader = get_mixup_loader(train_dataset, batch_size=batch_size,alpha=alpha)
    '''
    train_dataset = ClfSegDataset(crop_size=crop_size,move=random_move, subset=[1, 0, 4, 3])
    train_loader = get_loader(train_dataset, batch_size=batch_size)
    '''
        
    val_dataset = ClfvalSegDataset(crop_size=crop_size,move=None, subset=[2])
    val_loader = get_loader(val_dataset, batch_size=batch_size)
    
    test_dataset = ClfSegTestDataset(crop_size=crop_size,move=None)
    test_loader = get_loader_inorder(test_dataset, batch_size=1)

    model = densesharp.get_compiled(output_size=1,
                                    optimizer=Adam(lr=learning_rate),
                                    loss={"clf": 'binary_crossentropy',
                                          "seg": losses.DiceLoss()},
                                    metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure, metrics.auc],
                                             'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                    loss_weights={"clf": 1., "seg": segmentation_task_ratio},
                                    weight_decay=weight_decay,weights='tmp/test/weights.40_215456.h5')

    checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                                   period=1, save_weights_only=True)
    csv_logger = CSVLogger('tmp/%s/training.csv' % save_folder)
    tensorboard = TensorBoard(log_dir='tmp/%s/logs/' % save_folder)

    # 保存最好的一组weight
    best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
    
                                  
    # 防止过拟合 在max模式下，当检测值不再上升的时候则停止训练
    early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',
                                   patience=20, verbose=1)
    
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)
    
    # 使用model.fit_generator来节省内存
    '''
    
    model.fit_generator(generator=train_loader, steps_per_epoch=len(train_dataset), max_queue_size=100, workers=1,epochs=epochs,
                        callbacks=[checkpointer, lr_reducer, csv_logger, tensorboard])
    '''
    
    model.fit_generator(generator=train_loader, steps_per_epoch=50, max_queue_size=10, workers=1,
                    validation_data=val_loader, epochs=epochs, validation_steps=50,
                    callbacks=[checkpointer, csv_logger, best_keeper,early_stopping,lr_reducer,tensorboard])
    
    pred = model.predict_generator(generator=test_loader,steps=len(test_dataset), verbose=1)
    
    index = tuple(TEST.index)
    name = TEST.loc[index, 'name']
    name.tolist()
    
    
    df = pd.read_csv('Submission.csv',index_col=0)
    df['Id'] = name
    df['Predicted'] = pred[0]
    df.to_csv('Submission.csv')


if __name__ == '__main__':
    main(batch_size=32,
         crop_size=[32, 32, 32],
         random_move=3,
         learning_rate=1.e-5,
         segmentation_task_ratio=0.2,
         weight_decay=0.0,
         save_folder='test',
         epochs=200,
         alpha=1.0)
