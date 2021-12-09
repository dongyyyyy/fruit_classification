import torchvision.models as models
from models.SelectModel import *
from torchsummary import summary
from dataloader.dataloader import *
import random
import numpy as np
import torch
import csv
import torch.nn as nn
import sys
from tqdm import tnrange, tqdm
from utils.scheduler import *
import time
from torch.utils.data import DataLoader
import multiprocessing

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
def train(args):
    random.seed(args.random_seed)  # seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    gpu_num = 1
    train_files,val_files,test_files = read_dataset(args.data_path,split_index=args.split)

    save_path = f'./save_models/use_pretrained_{args.pretrained}_models_{args.model}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}/' \
                f'{args.split}_randomSeed_{args.random_seed}/' \
                f'imageSize_{args.size}_batchSize_{args.batch_size}/epochs_{args.epochs}_stop_{args.stop_epochs}/'\
                f'lossFunction_{args.loss_function}//'

    logging_path = f'./logging/use_pretrained_{args.pretrained}_models_{args.model}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}/' \
                f'{args.split}_randomSeed_{args.random_seed}/' \
                f'imageSize_{args.size}_batchSize_{args.batch_size}/epochs_{args.epochs}_stop_{args.stop_epochs}/'\
                f'lossFunction_{args.loss_function}//'

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(logging_path,exist_ok=True)


    logging_datasetfile = logging_path + 'datasetList.csv'
    with open(logging_datasetfile,'w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(train_files)
        writer.writerow(val_files)
        writer.writerow(test_files)


    logging_filename = logging_path + f'logging.txt'
    save_filename = save_path + 'save.pth'

    check_file = open(logging_filename, 'w')  # logging file

    model = select_model(model_name=args.model, pretrained=args.pretrained)
    # print(model.state_dict().keys())
    # exit(1)

    # EfficientNet
    model.classifier = nn.Linear(1280,6,bias=True)

    # ResNet
    # model.fc = nn.Linear(512,6,bias=False)

    if args.cuda == 0:
        device = torch.device('cpu')
        model = model.to(device)
    else:
        device = torch.device('cuda')
        model = model.to(device)


    logging_path = f''
    summary(model.cuda(),(3,224,224))

    # transforms_train = [transforms.Resize((args.size, args.size), Image.BICUBIC),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((mean), (std))]
    transforms_train = [transforms.Resize((args.size+30, args.size+30), Image.BICUBIC),
                        transforms.RandomCrop(args.size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((mean), (std))]
    transforms_val = [transforms.Resize((args.size, args.size), Image.BICUBIC),
                      transforms.ToTensor(),
                      transforms.Normalize((mean), (std))]

    train_dataset = fruit_dataloader(dataset_list=train_files,transforms_=transforms_train)

    val_dataset = fruit_dataloader(dataset_list=val_files,transforms_=transforms_val)
    cpu_num = multiprocessing.cpu_count()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                                  num_workers=(cpu_num // 4))

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=(cpu_num // 4))

    count = make_weights_for_balanced_classes(train_dataset.image_files_path)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-5,nesterov=False)
    elif args.optim =='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.5,0.999))
    elif args.optim == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(),lr=args.lr)
    elif args.optim =='adamW':
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.5,0.999))

    if args.loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif args.loss_function == 'CEW':
        beta = 0.99
        samples_per_cls = count / np.sum(count)
        no_of_classes = 6
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)

    if args.scheduler == 'WarmUp':
        print(f'target lr : {args.lr} / warmup_iter : {10}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - 5 + 1)
        scheduler = LearningRateWarmUP(optimizer=optimizer,
                                       warmup_iteration=5,
                                       target_lr=args.lr,
                                       after_scheduler=scheduler_cosine)
    elif args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'multiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [11, 21], gamma=0.1)
    elif args.scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.5, patience=10,
                                                               min_lr=1e-6)
    elif args.scheduler == 'Cosine':
        print('Cosine Scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
    best_accuracy = 0.
    stop_count = 0
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        train_total_loss = 0.0
        train_total_count = 0
        train_total_data = 0

        val_total_loss = 0.0
        val_total_count = 0
        val_total_data = 0

        start_time = time.time()
        model.train()

        output_str = 'current epoch : %d/%d / current_lr : %f \n' % (
        epoch + 1, args.epochs, optimizer.state_dict()['param_groups'][0]['lr'])
        sys.stdout.write(output_str)
        check_file.write(output_str)
        index = 0
        with tqdm(train_dataloader, desc='Train', unit='batch') as tepoch:
            for index, (batch_signal, batch_label) in enumerate(tepoch):
                batch_signal = batch_signal.to(device)
                batch_label = batch_label.long().to(device)

                optimizer.zero_grad()
                pred = model(batch_signal)

                # for parameter in model.parameters():
                #     norm += torch.norm(parameter, p=norm_square)

                loss = loss_fn(pred, batch_label)  # + beta * norm

                _, predict = torch.max(pred, 1)
                check_count = (predict == batch_label).sum().item()

                train_total_loss += loss.item()

                train_total_count += check_count
                train_total_data += len(predict)

                loss.backward()
                optimizer.step()
                accuracy = train_total_count / train_total_data
                tepoch.set_postfix(loss=train_total_loss / (index + 1), accuracy=100. * accuracy)

        train_total_loss /= index
        train_accuracy = train_total_count / train_total_data * 100

        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                     % (epoch + 1, args.epochs, time.time() - start_time, train_total_loss,
                        train_total_count, train_total_data, train_accuracy)
        # sys.stdout.write(output_str)
        check_file.write(output_str)

        # check validation dataset
        start_time = time.time()
        model.eval()
        index = 0
        with tqdm(val_dataloader, desc='Validation', unit='batch') as tepoch:
            for index, (batch_signal, batch_label) in enumerate(tepoch):
                batch_signal = batch_signal.to(device)

                batch_label = batch_label.long().to(device)

                with torch.no_grad():
                    pred = model(batch_signal)

                    loss = loss_fn(pred, batch_label)

                    # acc
                    _, predict = torch.max(pred, 1)
                    check_count = (predict == batch_label).sum().item()

                    val_total_loss += loss.item()

                    val_total_count += check_count
                    val_total_data += len(predict)

                    accuracy = val_total_count / val_total_data
                    tepoch.set_postfix(loss=val_total_loss / (index + 1), accuracy=100. * accuracy)

        val_total_loss /= index
        val_accuracy = val_total_count / val_total_data * 100

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                     % (epoch + 1, args.epochs, time.time() - start_time, val_total_loss,
                        val_total_count, val_total_data, val_accuracy)
        check_file.write(output_str)


        if epoch == 0:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_file = save_filename
            # torch.save(model.module.state_dict(), save_file)
            if gpu_num > 1:
                torch.save(model.module.state_dict(), save_file)
            else:
                torch.save(model.state_dict(), save_file)
            stop_count = 0
        else:
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                save_file = save_filename
                # torch.save(model.module.state_dict(), save_file)
                if gpu_num > 1:
                    torch.save(model.module.state_dict(), save_file)
                else:
                    torch.save(model.state_dict(), save_file)
                stop_count = 0
            else:
                stop_count += 1
        if stop_count >= args.stop_epochs:
            print('Early Stopping')
            break

        output_str = 'best epoch : %d/%d / val accuracy : %f%%\n' \
                     % (best_epoch + 1, args.epochs, best_accuracy)
        sys.stdout.write(output_str)
        print('=' * 30)

    output_str = 'best epoch : %d/%d / accuracy : %f%%\n' \
                 % (best_epoch + 1, args.epochs, best_accuracy)
    sys.stdout.write(output_str)
    check_file.write(output_str)
    print('=' * 30)

    check_file.close()


