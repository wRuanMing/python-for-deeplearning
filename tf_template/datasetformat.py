# -*- coding: utf-8 -*-
"""
Created on Sat May 19 10:16:22 2018

@author: wRuanMing
dataset transform 1->2 or 2->1
1.folder:e.g. Animal/cat/001.jpg,002.jpg... Animal/fish/008.jpg
2.imgscsv:e.g ./Animal/001.jpg,002.jpg... ./AnimalLabel.csv
"""
import os
import shutil
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', default='data', type = str)
    parser.add_argument('-t', '--tocsv', default= False, type = bool)
    return parser.parse_args()
def folder2imgscsv(datadir):
#    datadir = r'E:\MyCodeGit\tf_template\data\train'
    folder_imgname = [[os.path.basename(f),os.path.basename(dp)] for dp, dn, fn in os.walk(datadir) for f in fn 
                      if f.endswith('jpeg') or f.endswith('tiff') or f.endswith('png') or f.endswith('bmp')]
    df = pd.DataFrame(folder_imgname,columns = ['name','label'])
    csv_file = os.path.join(datadir,os.path.basename(datadir)+'_labels.csv')
    if not os.path.exists(csv_file):
        df.to_csv(csv_file,index=False)
    else:
        print("csv file already exists!")
    
    newimgs_folder = os.path.join(datadir,os.path.basename(datadir)+'_imgs')
    if not os.path.exists(newimgs_folder):
        os.mkdir(newimgs_folder)
    complete_imgpath = [os.path.join(dp, f) for dp, dn, fn in os.walk(datadir) for f in fn 
                        if f.endswith('jpeg') or f.endswith('tiff') or f.endswith('png') or f.endswith('bmp')]
    for img_path_name in complete_imgpath:
        shutil.move(img_path_name, newimgs_folder)
        
    for dp, dn, fn in os.walk(datadir):
        if fn == []:
            os.rmdir(dp)
def imgscsv2folder(datadir):
#    datadir = r'E:\MyCodeGit\tf_template\data\train'
    csv_file = os.path.join(datadir, os.path.basename(datadir) + '_labels.csv')
    csv_data = pd.read_csv(csv_file)
    imgs_folder = os.path.join(datadir, os.path.basename(datadir) + '_imgs')
    for img_name, img_label in csv_data.values:
        img_path_name = os.path.join(imgs_folder, img_name)
        folder_imglabel = os.path.join(datadir, img_label)
        if not os.path.exists(folder_imglabel):
            os.mkdir(folder_imglabel)
        shutil.move(img_path_name, folder_imglabel)   
        
    for dp, dn, fn in os.walk(datadir):
        if fn == []:
            os.rmdir(dp)

if __name__=='__main__':
    args = parse_args()
    print("dataset directory:{}".format(args.datadir))
    print("dataset transform：to csv：{} \t to folder：{}".format(args.tocsv,bool(True-args.tocsv)))
    print("transform start!")
    if args.tocsv==False:
        imgscsv2folder(args.datadir)
    else:
        folder2imgscsv(args.datadir)
    print("transform completed!")
    

