import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import argparse
import shutil
from random import shuffle
import math


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def set_file(data_dir_path, xml_file):
    xml_path = data_dir_path + '/' + xml_file
    txt_file = xml_file.replace('xml', 'txt')
    img_file = xml_file.replace('xml', 'jpg')
    img_path = data_dir_path + '/' + img_file
    return xml_path, txt_file, img_path
        
def convert_labels_annotation(data_dir_path, img_dir_path, label_dir_path, xml_list):
    for xml_file in xml_list:
        xml_path, txt_file, img_path = set_file(data_dir_path, xml_file)

        in_file = open(data_dir_path + '/' + xml_file)
        out_file = open(label_dir_path + '/' + txt_file, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        move_img_to_imgdir(xml_path, img_path, img_dir_path)



def move_img_to_imgdir(xml_path, img_path, img_dir_path):
    shutil.move(img_path, img_dir_path)
    os.remove(xml_path)       

def remove_unsed_img(data_dir_path):
    count = 0
    for img_file in glob.glob(data_dir_path + '/*.jpg'):
        print("unused image: ", img_file)
        os.remove(img_file)
        count += 1

    print("unused image: ", count)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='2020_Autumn', help='Select dataset')
    parser.add_argument('--split', type=float, default=0.7, help='percentage of splitting training and testing')
    args = parser.parse_args()
    return args

def initialize():
    args = get_args()
    cwd = getcwd()
    data_dir_path = cwd + '/' + args.dataset 
    train_test_dir = ['train', 'val']
    classes = ['Sorrel']
    split = args.split
    return data_dir_path, train_test_dir, classes, split

def create_image_label_dir(data_dir_path, train_test_dir):
    img_dir_path = data_dir_path + '/images/' + train_test_dir
    label_dir_path = data_dir_path + '/labels/' + train_test_dir
    # Check Directory
    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)
    if not os.path.exists(label_dir_path):
        os.makedirs(label_dir_path)
    return img_dir_path, label_dir_path

def get_trainset_and_testset(data_dir_path, split):
    xml_files_list = get_file_list_from_dir(data_dir_path)
    randomize_files(xml_files_list)
    train_xml_list, test_xml_list = split_to_train_test(xml_files_list, split)
    return train_xml_list, test_xml_list

def get_file_list_from_dir(data_dir_path):
    # Get a list of the files
    all_files = os.listdir(os.path.abspath(data_dir_path))
    xml_files_list = list(filter(lambda file: file.endswith('.xml'), all_files))
    return xml_files_list

# Randomize the files
def randomize_files(xml_files_list):
    shuffle(xml_files_list)

# Split files into training and testing dataset
def split_to_train_test(file_list, split=0.7):
    split_index = math.floor(len(file_list) * split)
    train_xml_list = file_list[:split_index]
    test_xml_list = file_list[split_index:]
    return train_xml_list, test_xml_list

def set_xml_list(train_test_dir, train_xml_list, test_xml_list):
    if train_test_dir == 'train':
        xml_list = train_xml_list
    else:
        xml_list = test_xml_list
    return xml_list

# Main
if __name__ == '__main__':

    data_dir_path, dirs, classes, split = initialize()
    train_img_list, test_img_list = get_trainset_and_testset(data_dir_path, split)

    for train_test_dir in dirs: 
        # train or test
        img_dir_path, label_dir_path = create_image_label_dir(data_dir_path, train_test_dir)
        xml_list = set_xml_list(train_test_dir, train_img_list, test_img_list)

        # Convert xml to txt
        # Move txt file to the /labels/ directory 
        convert_labels_annotation(data_dir_path, img_dir_path, label_dir_path, xml_list)

 
        print("Finished processing: " + train_test_dir)

    remove_unsed_img(data_dir_path)


