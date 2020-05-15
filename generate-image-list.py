#!/usr/bin/python
# -*-coding:utf-8-*-

#Generate absolute path list of images for 'FlyingThings3D -Sceneflow Dataset'

import os

def generate_image_list(data_dir= "C:/Users/chong/Downloads/data_scene_flow/training" , label_dir= "C:/Users\chong\Downloads\data_scene_flow"):
    left_data_files = []
    right_data_files = []
    label_files = []

    left_data_dir = os.path.join(data_dir,'image_2')
    right_data_dir = os.path.join(data_dir,'image_3')
    label_data_dir = os.path.join(data_dir,'disp_noc_0')
    f_train = open('train.lst', 'w')

    for file in os.listdir(left_data_dir):
        left_data_files.append(str(left_data_dir)  + '/' + str(file))
    for file in os.listdir(right_data_dir):
        right_data_files.append(str(right_data_dir)  + '/' + str(file))
    for file in os.listdir(label_data_dir):
        label_files.append(str(label_data_dir) + '/' + str(file))

    for left_data_file,right_data_file,label_file in zip(left_data_files,right_data_files,label_files):
        line = str(left_data_file) + '\t' + str(right_data_file) + '\t' + str(label_file) + '\n'
        f_train.write(line)
    f_train.close()

if __name__ == '__main__':
    generate_image_list()