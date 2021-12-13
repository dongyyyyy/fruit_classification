import torchvision.models as models
from models.SelectModel import *
from torchsummary import summary
from dataloader.dataloader import *
import random
import numpy as np
import torch
import csv
import torch.nn as nn
from torch.autograd import Variable
import sys
from tqdm import tnrange, tqdm
from utils.scheduler import *
import time
from torch.utils.data import DataLoader
import multiprocessing
from utils.dataAugmentation import *
from utils.loss_fn import *
import pandas as pd

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def test_folds(args):    
    csv_file_path = f'/mnt/hdd3/fruit_classification/csv/use_pretrained_{args.pretrained}_models_{args.model}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}/' \
                f'{args.split}_randomSeed_{args.random_seed}/' \
                f'imageSize_{args.size}_batchSize_{args.batch_size}/epochs_{args.epochs}_stop_{args.stop_epochs}/'\
                f'lossFunction_{args.loss_function}/'
    os.makedirs(csv_file_path,exist_ok=True)

    csv_file = csv_file_path + f'save_withoutNorm(0.0001)_randomCrop_fold({args.fold}).csv'
    confusion = torch.zeros(6, 6)
    random.seed(int(args.random_seed))  # seed
    np.random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    gpu_num = 4
    _,_,test_files = read_dataset(dataset_path=args.data_path,split_index=args.split)
    model_list = []
    for fold in range(int(args.fold)): 
        load_path = f'/mnt/hdd3/fruit_classification/save_models/final/use_pretrained_{args.pretrained}_models_{args.model}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}/' \
                    f'{args.split}_randomSeed_{args.random_seed}/' \
                    f'imageSize_{args.size}_batchSize_{args.batch_size}/epochs_{args.epochs}_stop_{args.stop_epochs}/'\
                    f'lossFunction_{args.loss_function}_kfold({args.fold})/'
        load_filename = load_path + f'save_withoutNorm(0.0001)_randomCrop_fold({args.fold}_{fold}).pth'
        model_list.append(select_model(model_name=args.model, pretrained=args.pretrained))
        model_list[fold].classifier = nn.Linear(1280,6,bias=True)
        model_list[fold].load_state_dict(torch.load(load_filename))
        if args.cuda == 0:
            device = torch.device('cpu')
            model_list[fold] = model_list[fold].to(device)
        else:
            device = torch.device('cuda')
            model_list[fold] = model_list[fold].to(device)
            if gpu_num >= 2:
                model_list[fold] = nn.DataParallel(model_list[fold])
        
        
        # print(model.state_dict().keys())
        # exit(1)

        # ResNet
        # model.fc = nn.Linear(512,6,bias=False)

       

    transforms_test = [transforms.Resize((int(args.size), int(args.size)), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((mean), (std))]

    test_dataset = fruit_dataloader(dataset_list=test_files,transforms_=transforms_test)
    cpu_num = multiprocessing.cpu_count()
    

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), pin_memory=True, num_workers=(cpu_num // 4))

    best_accuracy = 0.
    
        

    # check validation dataset
    start_time = time.time()
    for fold in range(int(args.fold)):
        model_list[fold].eval()
    index = 0
    test_total_count = 0
    test_total_data = 0
    with tqdm(test_dataloader, desc='Test', unit='batch') as tepoch:
        for index, (batch_signal, batch_label) in enumerate(tepoch):
            batch_signal = batch_signal.to(device)

            batch_label = batch_label.long().to(device)

            with torch.no_grad():
                for fold in range(int(args.fold)):
                    pred = model_list[fold](batch_signal)
                    if fold == 0:
                        predict = F.softmax(pred, 1)
                    else:
                        predict += F.softmax(pred,1)
                    # print(predict.shape)
                
                _, predict = torch.max(predict, 1)
                check_count = (predict == batch_label).sum().item()

                for index in range(len(batch_label)):
                    confusion[batch_label[index]][predict[index]] += 1

                test_total_count += check_count
                test_total_data += len(predict)

                accuracy = test_total_count / test_total_data
                tepoch.set_postfix(accuracy=100. * accuracy)
                

    test_accuracy = test_total_count / test_total_data * 100

    output_str = 'spend time : %.4f sec  /  correct : %d/%d -> %.4f%%\n' \
                % (time.time() - start_time,
                    test_total_count, test_total_data, test_accuracy)
    print(output_str)

    confusion_percent = torch.zeros(6, 6)
    confusion_percent_recall = torch.zeros(6, 6)
    sensitivity = torch.zeros(6)
    specificity = torch.zeros(6)
    recall = torch.zeros(6)
    precision = torch.zeros(6)

    tp = torch.zeros(6)
    tp_fp = confusion.sum(dim=0) # for precision
    tp_fn = confusion.sum(dim=1) # for recall
    tn_fp = torch.zeros(6)

    for i in range(6):
        for z in range(6):
            if z != i:
                tn_fp[i] += tp_fn[z]
            else:
                tp[i] += confusion[i][i]

    for index in range(6):
        confusion_percent[:,index] = confusion[:,index] / tp_fp.float()
        confusion_percent_recall[index,:] = confusion[index] / tp_fn.float()
        tn = confusion.sum() - confusion.sum(dim=0)[index] - confusion.sum(dim=1)[index] + confusion[index][index]
        sensitivity[index] = confusion[index][index] / float(tp_fn[index])
        specificity[index] = tn / float(tn_fp[index])
        precision[index] = confusion[index][index] / float(tp_fp[index])

        recall[index] = sensitivity[index]

    f1_score = 2 * (precision * recall) / (precision + recall)
    p_a = tp.sum() / confusion.sum()
    p_c = 0
    for i in range(6):
        p_c += (confusion.sum(dim=0)[i] / confusion.sum() * confusion.sum(dim=1)[i] / confusion.sum())
    kappa = (p_a-p_c) / (1-p_c)


    macro_f1_score = torch.mean(f1_score)
    macro_f1_score = macro_f1_score.numpy()
    kappa = kappa.numpy()
    sensitivity = sensitivity.numpy()
    specificity = specificity.numpy()
    precision = precision.numpy()
    recall = recall.numpy()
    f1_score = f1_score.numpy()

    balanced_accuracy = 0.
    for index in range(len(recall)):
        balanced_accuracy += recall[index]
    
    balanced_accuracy /= len(recall)

    # none / cohen's kappa, Balanced Accuracy / macro-f1-score / accuracy
    infomation = [0, kappa, macro_f1_score, balanced_accuracy*100, test_accuracy]

    df_list = [['','','','','',''],['','','','','',''],['','','','','',''],['','','','','',''],['','','','','',''],['','','','','','']]

    for (i1,j1), z1 in np.ndenumerate(confusion):
        for (i, j), z2 in np.ndenumerate(confusion_percent):
            # print(i, j, z)
            # plt.text(j, i, '%.2f\n' % z, ha='center', va='center', color='Red', fontSize=20, fontweight='bold')
            for (x, y), z3 in np.ndenumerate(confusion_percent_recall):
                # print(i, j, z)
                if i == x and x == i1 and j == y and y == j1:
                    df_list[i][j] = '%d\n(%.1f%%, %.1f%%)'%(z1,z2*100,z3*100)

    df = pd.DataFrame([specificity, recall, precision, f1_score, df_list[0],df_list[1],df_list[2],df_list[3],df_list[4],df_list[5],infomation],
                    columns=['정상', '궤양병', '귤응애', '진딧물', '점무늬병','총채벌레'],
                    index=[ 'specificity', 'recall', 'precision', 'f1_score','정상', '궤양병', '귤응애', '진딧물', '점무늬병','총채벌레','info'])

    df.to_csv(csv_file,mode='w',encoding='utf-8-sig',header=True)


            

def tests(args):
    csv_file_path = f'/mnt/hdd3/fruit_classification/csv/use_pretrained_{args.pretrained}_models_{args.model}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}/' \
                f'{args.split}_randomSeed_{args.random_seed}/' \
                f'imageSize_{args.size}_batchSize_{args.batch_size}/epochs_{args.epochs}_stop_{args.stop_epochs}/'\
                f'lossFunction_{args.loss_function}/'
    os.makedirs(csv_file_path,exist_ok=True)

    csv_file = csv_file_path + 'save_withoutNorm(0.0001)_randomCrop.csv'

    load_path = f'/mnt/hdd3/fruit_classification/save_models/final/use_pretrained_{args.pretrained}_models_{args.model}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}/' \
                f'{args.split}_randomSeed_{args.random_seed}/' \
                f'imageSize_{args.size}_batchSize_{args.batch_size}/epochs_{args.epochs}_stop_{args.stop_epochs}/'\
                f'lossFunction_{args.loss_function}/'
    load_filename = load_path + 'save_withoutNorm(0.0001)_randomCrop.pth'
    confusion = torch.zeros(6, 6)

    random.seed(int(args.random_seed))  # seed
    np.random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    gpu_num = 4
    _,_,test_files = read_dataset(dataset_path=args.data_path,split_index=args.split)
    
    model = select_model(model_name=args.model, pretrained=args.pretrained)
    # print(model.state_dict().keys())
    # exit(1)

    # EfficientNet
    model.classifier = nn.Linear(1280,6,bias=True)
    model.load_state_dict(torch.load(load_filename))
    # ResNet
    # model.fc = nn.Linear(512,6,bias=False)

    if args.cuda == 0:
        device = torch.device('cpu')
        model = model.to(device)
    else:
        device = torch.device('cuda')
        model = model.to(device)
        if gpu_num >= 2:
            model = nn.DataParallel(model)

    transforms_test = [transforms.Resize((int(args.size), int(args.size)), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((mean), (std))]

    test_dataset = fruit_dataloader(dataset_list=test_files,transforms_=transforms_test)
    cpu_num = multiprocessing.cpu_count()
    

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), pin_memory=True, num_workers=(cpu_num // 4))

    best_accuracy = 0.
    
        

    # check validation dataset
    start_time = time.time()
    model.eval()
    index = 0
    test_total_count = 0
    test_total_data = 0
    with tqdm(test_dataloader, desc='Test', unit='batch') as tepoch:
        for index, (batch_signal, batch_label) in enumerate(tepoch):
            batch_signal = batch_signal.to(device)

            batch_label = batch_label.long().to(device)

            with torch.no_grad():
                pred = model(batch_signal)

                # acc
                _, predict = torch.max(pred, 1)
                check_count = (predict == batch_label).sum().item()

                for index in range(len(batch_label)):
                    confusion[batch_label[index]][predict[index]] += 1

                test_total_count += check_count
                test_total_data += len(predict)

                accuracy = test_total_count / test_total_data
                tepoch.set_postfix(accuracy=100. * accuracy)

    test_accuracy = test_total_count / test_total_data * 100

    output_str = 'spend time : %.4f sec  /  correct : %d/%d -> %.4f%%\n' \
                % (time.time() - start_time,
                    test_total_count, test_total_data, test_accuracy)
    print(output_str)

    confusion_percent = torch.zeros(6, 6)
    confusion_percent_recall = torch.zeros(6, 6)
    sensitivity = torch.zeros(6)
    specificity = torch.zeros(6)
    recall = torch.zeros(6)
    precision = torch.zeros(6)

    tp = torch.zeros(6)
    tp_fp = confusion.sum(dim=0) # for precision
    tp_fn = confusion.sum(dim=1) # for recall
    tn_fp = torch.zeros(6)

    for i in range(6):
        for z in range(6):
            if z != i:
                tn_fp[i] += tp_fn[z]
            else:
                tp[i] += confusion[i][i]

    for index in range(6):
        confusion_percent[:,index] = confusion[:,index] / tp_fp.float()
        confusion_percent_recall[index,:] = confusion[index] / tp_fn.float()
        tn = confusion.sum() - confusion.sum(dim=0)[index] - confusion.sum(dim=1)[index] + confusion[index][index]
        sensitivity[index] = confusion[index][index] / float(tp_fn[index])
        specificity[index] = tn / float(tn_fp[index])
        precision[index] = confusion[index][index] / float(tp_fp[index])

        recall[index] = sensitivity[index]

    f1_score = 2 * (precision * recall) / (precision + recall)
    p_a = tp.sum() / confusion.sum()
    p_c = 0
    for i in range(6):
        p_c += (confusion.sum(dim=0)[i] / confusion.sum() * confusion.sum(dim=1)[i] / confusion.sum())
    kappa = (p_a-p_c) / (1-p_c)


    macro_f1_score = torch.mean(f1_score)
    macro_f1_score = macro_f1_score.numpy()
    kappa = kappa.numpy()
    sensitivity = sensitivity.numpy()
    specificity = specificity.numpy()
    precision = precision.numpy()
    recall = recall.numpy()
    f1_score = f1_score.numpy()

    balanced_accuracy = 0.
    for index in range(len(recall)):
        balanced_accuracy += recall[index]
    
    balanced_accuracy /= len(recall)

    # none / cohen's kappa, Balanced Accuracy / macro-f1-score / accuracy
    infomation = [0, kappa, macro_f1_score, balanced_accuracy*100, test_accuracy]

    df_list = [['','','','','',''],['','','','','',''],['','','','','',''],['','','','','',''],['','','','','',''],['','','','','','']]

    for (i1,j1), z1 in np.ndenumerate(confusion):
        for (i, j), z2 in np.ndenumerate(confusion_percent):
            # print(i, j, z)
            # plt.text(j, i, '%.2f\n' % z, ha='center', va='center', color='Red', fontSize=20, fontweight='bold')
            for (x, y), z3 in np.ndenumerate(confusion_percent_recall):
                # print(i, j, z)
                if i == x and x == i1 and j == y and y == j1:
                    df_list[i][j] = '%d\n(%.1f%%, %.1f%%)'%(z1,z2*100,z3*100)

    df = pd.DataFrame([specificity, recall, precision, f1_score, df_list[0],df_list[1],df_list[2],df_list[3],df_list[4],df_list[5],infomation],
                    columns=['정상', '궤양병', '귤응애', '진딧물', '점무늬병','총채벌레'],
                    index=[ 'specificity', 'recall', 'precision', 'f1_score','정상', '궤양병', '귤응애', '진딧물', '점무늬병','총채벌레','info'])

    df.to_csv(csv_file,mode='w',encoding='utf-8-sig',header=True)