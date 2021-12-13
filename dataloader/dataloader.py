import torch
import os
import torchvision.transforms as transforms
import json
import random
from PIL import Image


def make_weights_for_balanced_classes(data_list, nclasses=6):
    count = [0] * nclasses

    for data in data_list:
        jsonfile = '/'.join(data.split('.')[:-1]) + '.json'
        with open(jsonfile, 'r', encoding='UTF8') as json_file:
            json_data = json.load(json_file)
            annotations = json_data["Annotations"]["OBJECT_CLASS_CODE"]
            if annotations == '감귤_정상' or annotations == '키위_정상':
                labels = 0
            elif annotations == '감귤_궤양병' or annotations == '키위_궤양병':
                labels = 1
            elif annotations == '감귤_귤응애':
                labels = 2
            elif annotations == '감귤_진딧물':
                labels = 3
            elif annotations == '키위_점무늬병':
                labels = 4
            elif annotations == '키위_총채벌레':
                labels = 5
        count[labels] += 1

    return count

class fruit_dataloader(object):
    def __init__(self,
                 dataset_list,
                 transforms_=None,
                 ):

        self.transform = transforms.Compose(transforms_)
        self.image_files_path = dataset_list

    def __getitem__(self, index):
        image = self.transform(Image.open(self.image_files_path[index]))
        jsonfile = '/'.join(self.image_files_path[index].split('.')[:-1]) + '.json'
        with open(jsonfile, 'r', encoding='UTF8') as json_file:
            json_data = json.load(json_file)
            annotations = json_data["Annotations"]["OBJECT_CLASS_CODE"]
            if annotations == '감귤_정상' or annotations == '키위_정상':
                labels=0
            elif annotations == '감귤_궤양병' or annotations == '키위_궤양병':
                labels=1
            elif annotations == '감귤_귤응애':
                labels=2
            elif annotations == '감귤_진딧물':
                labels=3
            elif annotations == '키위_점무늬병':
                labels=4
            elif annotations == '키위_총채벌레':
                labels=5

        return image, labels

    def __len__(self):
        return len(self.image_files_path)

def read_dataset(dataset_path,split_index=0.7):
    train_files = []
    val_files = []
    test_files = []
    dataset_list = os.listdir(dataset_path)

    for dataset_folder in dataset_list:
        signals_path = dataset_path+dataset_folder+'/'
        signals_list = os.listdir(signals_path)
        random.shuffle(signals_list)
        length = len(signals_list)//2
        train_length = int(length*split_index)
        val_length = int(length*((1-split_index)/2))
        index = 0
        for signals_filename in signals_list:
            if signals_filename.split('.')[-1] == 'jpg':
                signals_file = signals_path + signals_filename
                if train_length > index:
                    train_files.append(signals_file)
                elif train_length+val_length > index:
                    val_files.append(signals_file)
                else:
                    test_files.append(signals_file)
                index += 1

    return train_files,val_files,test_files

def read_fold_dataset(dataset_path,fold_num,max_fold,split_index=0.7):
    train_files = []
    val_files = []
    test_files = []
    dataset_list = os.listdir(dataset_path)
    
    for dataset_folder in dataset_list:
        signals_path = dataset_path+dataset_folder+'/'
        signals_list = os.listdir(signals_path)
        random.shuffle(signals_list)
        length = len(signals_list)//2
        train_length = int(length*split_index)
        val_length = int(length*((1-split_index)/2))
        # print(f'kfold_length = {(train_length + val_length)//max_fold} // max_fold = {max_fold} // current_fold = {fold_num}')
        fold_length = (train_length + val_length)//max_fold

        index = 0
        for signals_filename in signals_list:
            if signals_filename.split('.')[-1] == 'jpg':
                signals_file = signals_path + signals_filename
                if train_length+val_length > index:
                    if fold_num*fold_length <= index and (fold_num+1)*fold_length > index:
                        val_files.append(signals_file)
                    else:
                        train_files.append(signals_file)
                else:
                    test_files.append(signals_file)
                index += 1
    return train_files,val_files,test_files

