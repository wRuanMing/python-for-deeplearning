# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:22:37 2018

@author: BigVision
"""
import argparse
#import pandas as pd
#import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdir', default='data\\data_tfrecord\\test\\', type = str)
    parser.add_argument('--traindir', default='data\\data_tfrecord\\train\\', type = str)
    parser.add_argument('--valdir', default='data\\data_tfrecord\\train\\', type = str)
    parser.add_argument('--datainfo', default='data\\data_tfrecord\\data_info.csv', type = str)
    parser.add_argument('--batchsize', default=1, type = int)
    parser.add_argument('--epochs', default=2, type = int)
    parser.add_argument('--epochs_between_evals', default=1, type = int)
#    parser.add_argument('--trainortest', default='train', type = str)
    parser.add_argument('--imgsize', default=128, type=int)
    parser.add_argument('--imgchannel', default=3, type=int)
    return parser.parse_args()
tfr = TFrecorder()
def input_fn_maker(path, data_info_path, shuffle=False, batch_size = 1, epoch = 3, padding = None):
    def input_fn():
        # tfr.get_filenames会返回包含path下的所有tfrecord文件的list
        # shuffle会让这些文件的顺序打乱
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle, 
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn
def model_fn(features, mode):
    # reshape 784维的图片到28x28的平面表达，1为channel数
    features['image'] = tf.reshape(features['image'],[-1,128,128,3])
    # shape: [None,512,512,1]
    conv1 = tf.layers.conv2d(
        inputs=features['image'],
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv1')
    # shape: [None,512,512,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name= 'pool1')
    # shape: [None,256,256,32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv2')
    # shape: [None,256,256,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name= 'pool2')
    # shape: [None,128,128,64]
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64], name= 'pool2_flat')
    # shape: [None,1048576]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name= 'dense1')
    # shape: [None,1024]
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # shape: [None,1024]
    logits = tf.layers.dense(inputs=dropout, units=10, name= 'output')
    # shape: [None,10]
    predictions = {
        "image":features['image'],
        "conv1_out":conv1,
        "pool1_out":pool1,
        "conv2_out":conv2,
        "pool2_out":pool2,
        "pool2_flat_out":pool2_flat,
        "dense1_out":dense1,
        "logits":logits,
        "classes": tf.argmax(input=logits, axis=1),
        "labels": features['label'],
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=features['label'], logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=features['label'], predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def main_train_and_test_model(command_args):
    padding_info = ({'image':[command_args.imgchannel*command_args.imgsize**2,],'label':[]})
    # 生成3个input_fn
    test_input_fn = input_fn_maker(command_args.testdir,  command_args.datainfo,
                                   padding = padding_info)
    train_input_fn = input_fn_maker(command_args.traindir,  command_args.datainfo, shuffle=True, batch_size = command_args.batchsize, 
                                    epoch = command_args.epochs_between_evals, padding = padding_info)
    # 用来评估训练集用，不想要shuffle
    train_eval_fn = input_fn_maker(command_args.valdir,  command_args.datainfo, batch_size = command_args.batchsize, padding = padding_info)
    # input_fn在执行时会返回一个字典，里面的key对应着不同的feature(包括label在内)
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor",
                     "probabilities": "softmax_tensor",}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="model")
    
    
    for _ in range(command_args.epochs // command_args.epochs_between_evals):
        mnist_classifier.train(
        input_fn=train_input_fn)
        eval_results = mnist_classifier.evaluate(input_fn=train_eval_fn)
        print('val set')
        print(eval_results)
        
    predicts_results = mnist_classifier.predict(input_fn=test_input_fn)
    print('test set')
    print(predicts_results)
#        eval_results = mnist_classifier.evaluate(input_fn=test_input_fn)
#        print('test set')
#        print(eval_results)
#    predicts =list(mnist_classifier.predict(input_fn=test_input_fn))
    
#    print(predicts[0].keys())
#    print(predicts[0]['image'].shape)
    
#    plt.imshow(predicts[0]['image'][:,:,:],cmap = plt.cm.gray)

if __name__ == '__main__':
    args = parse_args()
    main_train_and_test_model(args)