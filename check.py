from os import listdir, getcwd
from os.path import isfile, join
import argparse
import sys
import glob
import shutil
import os
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='draw_dataset', help='Select dataset')
    args = parser.parse_args()
    return args

def get_label_train_val_dir():
    args = get_args()
    cwd = getcwd()
    data_dir_path = cwd + '/' + args.dataset + '/labels' 
    train_label_dir = data_dir_path + '/train'
    val_label_dir = data_dir_path + '/val'

    return train_label_dir, val_label_dir

def get_img_train_val_dir():
    args = get_args()
    cwd = getcwd()
    data_dir_path = cwd + '/' + args.dataset + '/images'
    train_img_dir = data_dir_path + '/train'
    val_img_dir = data_dir_path + '/val'
    error_train_img_dir = data_dir_path + '/error_train'
    error_val_img_dir = data_dir_path + '/error_val'


    return train_img_dir, val_img_dir, error_train_img_dir, error_val_img_dir

if __name__ == '__main__':
    eps = sys.float_info.epsilon
    eps1 = 0.0005
    train_label_dir, val_label_dir = get_label_train_val_dir()
    train_img_dir, val_img_dir, error_train_img_dir, error_val_img_dir = get_img_train_val_dir()

    train_lable_files = [f for f in listdir(train_label_dir) if isfile(join(train_label_dir, f))]
    val_lable_files = [f for f in listdir(val_label_dir) if isfile(join(val_label_dir, f))]
    
    print("checking train file")
    for train_label_file in train_lable_files:
        with open(train_label_dir + '/' + train_label_file,"r") as f:
            #train_img_file = train_label_file.replace('txt', 'jpg')
            #image = cv2.imread(train_img_dir + '/' + train_img_file)
            error_img = False
            #height, width = image.shape[:2]
            height = 1080
            width = 1920
            for line in f:
                number = line.split()
                x = float(number[1])
                y = float(number[2])
                w = float(number[3])
                h = float(number[4])
              

                x1 = int((x - w/2) * width) 
                y1 = int((y - h/2) * height)
                x2 = int((x + w/2) * width)
                y2 = int((y + h/2) * height)



                #print("x1: ", x1, " y1: ", y1," x2: ", x2, " y2: ", y2)

                #cv2.rectangle(image,(x1, y1) , (x2 ,y2) ,(0,255,0),3)
                if abs(x2 - x1) <= 10  or  abs(y2 - y1) <= 10:   
                    print("error: ", train_label_file)
                    print("x1: ", x1, " y1: ", y1," x2: ", x2, " y2: ", y2)
                    #train_img_file = train_label_file.replace('txt', 'jpg')
                    #shutil.copy(train_img_dir + '/' + train_img_file, error_train_img_dir)          
                    #image = cv2.imread(error_train_img_dir + '/' + train_img_file)
                    #height, width = image.shape[:2]
                    #print("h: ", height, " w: ", width)
                    #cv2.rectangle(image,(int(x1 * width), int(y1 * height)),(int(x2 * width), int(y2 *height)),(0,255,0),3)
                    #error_img = True

            #if error_img == True:        
            #    cv2.imwrite(error_train_img_dir + '/' + train_img_file, image)   
            #else:
            #    cv2.imwrite(train_img_dir + '/' + train_img_file, image)



    print("checking val file") 
    for val_label_file in val_lable_files:
        with open(val_label_dir + '/' + val_label_file,"r") as f:
            for line in f:
                number = line.split()
                x1 = float(number[1])
                y1 = float(number[2])
                x2 = float(number[3])
                y2 = float(number[4])
    
                x1 = int((x - w/2) * width)
                y1 = int((y - h/2) * height)
                x2 = int((x + w/2) * width)
                y2 = int((y + h/2) * height)



                if abs(x2 - x1) <= 10  or  abs(y2 - y1) <= 10:
                    print("error: ", train_label_file)
                    print("x1: ", x1, " y1: ", y1," x2: ", x2, " y2: ", y2)


    print("finish checking")
