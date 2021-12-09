import argparse
from train import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',required=False,help='input model name',default='resnet18')
    parser.add_argument('-pretrained',required=False,help='using pretrained model',default=True)
    parser.add_argument('-mode',required=False,help='train & test',default='None')
    parser.add_argument('-data_path',required=False,help='data path',default='C:/fruit/')
    parser.add_argument('-random_seed',required=False,help='random_seed',default=0)

    parser.add_argument('-split',required=False,help='split dataset',default=0.7)

    parser.add_argument('-size',required=False,help='image resize',default=224)
    parser.add_argument('-batch_size',required=False,help='mini-batch size',default=64)
    parser.add_argument('-epochs',required=False,help='epochs',default=100)
    parser.add_argument('-stop_epochs',required=False,help='stop epochs',default=5)
    parser.add_argument('-scheduler',required=False,help='learning rate scheduler',default='StepLR')
    parser.add_argument('-loss_function',required=False,help='loss function',default='CE')

    parser.add_argument('-optim', required=False, help='optimizer', default='sgd')
    parser.add_argument('-lr', required=False, help='learning rate', default=0.1)
    parser.add_argument('-augmentation', required=False, help='data augmentation', default=True)

    parser.add_argument('-cuda',required=False,help='cpu(0) or gpu(1)',default=1)
    parser.add_argument('-gpu',required=False,help='gpu num',default=0)
    args = parser.parse_args()
    if args.mode =='None':
        print(args)
        print('Please add arg -mode train or test!!!')
    elif args.mode =='train':
        print('Train mode')
        train(args)
    elif args.mode =='test':
        print('Test mode')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
