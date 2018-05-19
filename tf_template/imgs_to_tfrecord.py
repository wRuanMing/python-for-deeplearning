# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:45:49 2018

@author: vcc
"""

# 所需库包
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tfrecorder import TFrecorder
from PIL import Image
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
#import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--istrain', default= True, type = bool)
    parser.add_argument('-d', '--datadir', default= 'data', type = str)
    parser.add_argument('--imgsize', default=512, type=int)
    parser.add_argument('--imgsnumperfile', default=1000, type=int)
    return parser.parse_args()
def get_imgsnameandlabels_from_dataset_folder(directory, label_digitization = True):
    #输入图像文件夹路径，输出文件夹中图像的所有图像名称+标签（子文件夹名称）
    imgs_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
    for f in fn if f.endswith('jpeg') or f.endswith('tiff') or f.endswith('png') or f.endswith('bmp')]
    labels_list = [file_temp.split('\\')[-2] for file_temp in imgs_list]
    if label_digitization == False:
        img_dict = {"img_name":imgs_list,"img_label":labels_list}
    else:
        label_nametonum = {}
        for index, label_name in enumerate(os.listdir(directory)):           
            label_nametonum[label_name] = index
        img_dict = {"img_name":imgs_list,"img_label":list(map(lambda x:label_nametonum[x],labels_list))}
    return img_dict
def preprocessed_imgs_to_tfrecord(command_args):
    if command_args.istrain:
        transfer_dir = os.path.join(command_args.datadir, 'train')
    else:
        transfer_dir = os.path.join(command_args.datadir, 'test')
    imgs_dict = get_imgsnameandlabels_from_dataset_folder(transfer_dir)        
    ## mnist数据
    ##mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # 指定如何写成tfrecord文件的信息
    # 每一个row是一个feature
    df = pd.DataFrame({'name':['image','label'],
                      'type':['float32','int64'],
                      'shape':[(command_args.imgsize**2,),()],
                      'isbyte':[False,False],
                      "length_type":['fixed','fixed'],
                      "default":[np.NaN,np.NaN]})
    # 实例化该类
    tfr = TFrecorder()
    dataset_record_name = 'data_tfrecord'
    if not os.path.exists(os.path.join(command_args.datadir, dataset_record_name)):
        os.mkdir(os.path.join(command_args.datadir, dataset_record_name))
    if not os.path.exists(os.path.join(command_args.datadir, dataset_record_name,transfer_dir.split('\\')[-1])):
        os.mkdir(os.path.join(command_args.datadir, dataset_record_name, transfer_dir.split('\\')[-1]))
    
    # 用该方法写训练集的tfrecord文件
    #dataset = mnist.train
    path = os.path.join(command_args.datadir, dataset_record_name, transfer_dir.split('\\')[-1], transfer_dir.split('\\')[-1])

    # 每个tfrecord文件写多少个样本
    num_examples_per_file = command_args.imgsnumperfile
    # 当前写的样本数
    num_so_far = 0
    # 要写入的文件
    writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, num_examples_per_file))
    # 写多个样本
    #for i in np.arange(dataset.num_examples):
    for i in np.arange(len(imgs_dict["img_name"])):
        # 要写到tfrecord文件中的字典
        features = {}
        # 写一个样本的图片信息存到字典features中
        tfr.feature_writer(df.iloc[0], Image.open(imgs_dict["img_name"][i]).resize((512,512)).tobytes(), features)
        # 写一个样本的标签信息存到字典features中
        tfr.feature_writer(df.iloc[1], imgs_dict["img_label"][i], features)
        
        tf_features = tf.train.Features(feature= features)
        tf_example = tf.train.Example(features = tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        print(i)
        # 每写了num_examples_per_file个样本就令生成一个tfrecord文件
        if i%num_examples_per_file ==0 and i!=0:
            writer.close()
            num_so_far = i
            writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
            print('saved %s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
    writer.close()
    # 把指定如何写成tfrecord文件的信息保存起来
    data_info_path = os.path.dir(command_args.datadir, dataset_record_name,'data_info.csv')
    df.to_csv(data_info_path,index=False)

if __name__ == '__main__':
    args = parse_args()
    preprocessed_imgs_to_tfrecord(args)