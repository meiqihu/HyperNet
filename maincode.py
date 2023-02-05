# pytorch code of TGRS paper
# "HyperNet: Self-Supervised Hyperspectral SpatialSpectral Feature Understanding Network for Hyperspectral Change Detection"

import os
import torch.nn as nn
import random
import math
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from HyperNet_model import HyperNet, BasicBlock
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_num_threads(2)



def initNetParams_v2(net):
    # Init net parameters
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def get_args_viareggio_EX_1(seed):
    print('---------------------------func: get_args_viareggio_EX_1---------------------------')
    print('---------------------------EX need be a number to set seed ------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Viareggio_data.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='EX_1', type=str, help='China dataset use img_1 and img_2 as input;')
    parser.add_argument('--idx_file', default='./data/num_idx_ex1.mat',type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--save_model_path',default=path + '/EX1_Viareggio_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',default=path + '/EX1_Viareggio_result' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--GT_ex1', default=r'/data/meiqi.hu/PycharmProjects/ACD/data/ref_EX1.bmp',
                        type=str, help='ground_truth map for EX-1')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args


def get_args_viareggio_EX_2(seed):
    print('---------------------------func: get_args_viareggio_EX_2---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Viareggio_data.mat', type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='EX_2', type=str, help='China dataset use img_1 and img_2 as input;')
    parser.add_argument('--idx_file',default='./data/num_idx_ex2.mat',type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--save_model_path',default=path + '/EX2_Viareggio_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',default=path + '/EX2_Viareggio_result' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--GT_ex2', default=r'/data/meiqi.hu/PycharmProjects/ACD/data/ref_EX2.bmp',
                        type=str, help='ground_truth map for EX-2')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)

    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)

    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args


def get_args_simulation_EX_3(seed):
    print('---------------------------func: get_args_simulation_EX_3---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Hymap.mat', type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='EX_3', type=str, help='EX_1 use img_1 and img_2_RE as input; EX_2 use img_1 and img_3_RE as input')
    parser.add_argument('--idx_file', default='./data/num_idx_ex3.mat', type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--save_model_path', default=path+'/EX3_Viareggio_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path+'/EX3_Viareggio_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)

    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)

    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args



def train_HyperNet_HACD(args):
    print('---------------------------func: train_HyperNet_HACD---------------------------')
    model, idx, img_2, groundTruth = [], [], [], []
    print('\n')
    data = sio.loadmat(args.data_name)
    print('input data for test:', args.data_name, sep='\n')
    img_1 = data['img_1']
    if args.EX_num == 'EX_1':
        model = HyperNet(BasicBlock, layernum=[127, 64, 128, 64], gamma=args.gamma)  # for EX-1 & EX-2
        img_2 = data['img_2_RE']  # H,W,# C
        print('img_1 and img_2_RE is input for test')
        idx = sio.loadmat(args.idx_file)['idx_EX1']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        groundTruth = Image.open(args.GT_ex1)

    elif args.EX_num == 'EX_2':
        model = HyperNet(BasicBlock, layernum=[127, 64, 128, 64], gamma=args.gamma)  # for EX-1 & EX-2
        img_2 = data['img_3_RE']  # H,W,C
        print('img_1 and img_3_RE is input for test')
        idx = sio.loadmat(args.idx_file)['idx_EX2']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        groundTruth = Image.open(args.GT_ex2)

    elif args.EX_num == 'EX_3':
        model = HyperNet(BasicBlock, layernum=[126, 64, 128, 64], gamma=args.gamma)  # for EX-3
        img_2 = data['img_2']  # H,W,C
        print('img_1 and img_2 is input for test')
        idx = sio.loadmat(args.idx_file)['idx_EX3']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
        groundTruth = data['GT']

    # H, W, C = img_1.shape
    X1 = torch.tensor(np.transpose(img_1, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    X2 = torch.tensor(np.transpose(img_2, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    print('input.shape:', X1.shape)
    del img_1, img_2, data

    model.apply(initNetParams_v2)
    init_lr = args.lr
    model.cuda()
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    Tra_ls = []
    print('trainging begins----------------------------')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        loss = model(X1, X2, idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Tra_ls.append(loss.item())
        if epoch % 10 == 0:
            print('epoch [{}/{}],train:{:.4f}'.format(epoch, args.epochs, loss.item()))
    print('--------------SSL: training is sucessfully down--------------')
    print('--------------model save_path:', args.save_model_path, sep='\n')
    # torch.save(model.state_dict(), args.save_model_path)

    plt.figure(args.EX_num)
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(args.epochs), np.asarray(Tra_ls), 'r-o', label="SSL")
    plt.legend()

    model.eval()
    print('---------------------- fuse feature---------------------')
    f1, f2 = model(X1, X2, 0)  # [1, 32, 450, 375]
    f1, f2 = f1.squeeze(), f2.squeeze()
    f1, f2 = np.array(f1.permute(1, 2, 0)), np.array(f2.permute(1, 2, 0))  # 450, 375,32]

    r = diff_RX(f1, f2)
    x, y, auc = plot_roc(r, groundTruth)
    plt.figure(args.EX_num + ' SSL fuse feature')
    plt.imshow(r, cmap='hot')

    # sio.savemat(args.save_result_path, {'r': r, 'x': x, 'y': y, 'auc': auc})
    # print('--------------result save_path:', args.save_result_path, sep='\n')

    return r, x, y, auc



def get_args_USA(seed):
    print('---------------------------func: get_args_USA---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/USA.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='USA',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file',
                        default='./data/num_idx_usa.mat',
                        type=str, help='path filename of the trained model')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')
    parser.add_argument('--save_model_path',default=path + '/USA_HyperNet_' + str(seed) + '.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path', default=path + '/USA_HyperNet_result_' + str(seed) + '.mat',
                        type=str, help='path filename of the trained model')


    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)

    return args


def get_args_Bay(seed):
    print('---------------------------func: get_args_Bay---------------------------')

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--data_name', default='./data/Bay.mat',
                        type=str, help='path filename of training data')
    parser.add_argument('--EX_num', default='Bay',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--idx_file', default='./data/num_idx_Bay.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    parser.add_argument('--save_model_path',
                        default= path + '/Bay_HyperNet_'+str(seed)+'.pkl',
                        type=str, help='path filename of the trained model')
    parser.add_argument('--save_result_path',
                        default= path + '/Bay_HyperNet_result'+str(seed)+'.mat',
                        type=str, help='path filename of the trained model')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args


def get_args_Barbara(seed, mode):
    print('---------------------------func: get_args_Barbara---------------------------')
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--EX_num', default='Barbara',
                        type=str, help='use img_1 and img_2_RE as input')
    parser.add_argument('--seed', default=seed, type=int, metavar='seed', help='seed for randn seed')

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        print('There is ', path)

    if mode == 'Barbara_half1':
        parser.add_argument('--data_name',default='./data/Barbara_half1.mat',
                            type=str, help='path filename of training data')
        parser.add_argument('--idx_file',default='./data/num_idx_Barbara_half1.mat',
                            type=str, help='path filename of the trained model')
        parser.add_argument('--save_model_path', default=path +'/Barbara_half1_HyperNet_' + str(seed) + '.pkl',
                            type=str, help='path filename of the trained model')
        parser.add_argument('--save_result_path', default=path +'/Barbara_half1_HyperNet_result' + str(seed) + '.mat',
                            type=str, help='path filename of the trained model')
    elif mode == 'Barbara_half2':
        parser.add_argument('--data_name', default='./data/Barbara_half2.mat',
                            type=str, help='path filename of training data')
        parser.add_argument('--idx_file', default='./data/num_idx_Barbara_half2.mat',
                            type=str, help='path filename of the trained model')

        parser.add_argument('--save_model_path', default=path + '/Barbara_half2_HyperNet_' + str(seed) + '.pkl',
                            type=str, help='path filename of the trained model')
        parser.add_argument('--save_result_path', default=path + '/Barbara_half2_HyperNet_result' + str(seed) + '.mat',
                            type=str, help='path filename of the trained model')
    else:
        print('-----mode should be Barbara_half1 or Barbara_half2-----------')

    parser.add_argument('--gamma', default=1, type=float, metavar='gamma', help='gamma for Focal_Cos')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    print('Runing:  ', args.EX_num)
    print('saved model path:', args.save_model_path)
    print('saved result path:', args.save_result_path)
    print('input data:  ', args.data_name)
    print('training idx_file:', args.idx_file)
    print('epochs:', args.epochs)
    print('seed:  ', args.seed)
    return args


def train_HyperNet_HBCD(args):
    print('---------------------------func: train_HyperNet_HBCD---------------------------')
    print('\n')
    model, idx = [], []
    if args.EX_num == 'USA':
        model = HyperNet(BasicBlock, layernum=[154, 72, 144, 72], gamma=args.gamma)  # for USA
        idx = sio.loadmat(args.idx_file)['idx_usa']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)

    elif args.EX_num == 'Bay':
        model =HyperNet(BasicBlock,  layernum=[224, 112, 224, 112], gamma=1)  # for Bay area dataset
        idx = sio.loadmat(args.idx_file)['idx_bay']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)
    elif args.EX_num == 'Barbara':
        print('------------training for Barbara dataset ------------------')
        model = HyperNet(BasicBlock, layernum=[224, 112, 224, 112], gamma=1)  # for Santa Barbara dataset
        idx = sio.loadmat(args.idx_file)['idx_barbara']
        idx = torch.from_numpy(idx.squeeze()).cuda()
        print('unchanged idx path:', args.idx_file)

    model.apply(initNetParams_v2)
    print('----------------model.apply(initNetParams_v2)-----------------')

    data = sio.loadmat(args.data_name)
    print('input data for test:', args.data_name, sep='\n')
    img_1 = data['img_1']
    img_2 = data['img_2']  # H,W,C
    print('img_1 and img_2 is input for test')
    H, W, C = img_1.shape
    X1 = torch.tensor(np.transpose(img_1, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    X2 = torch.tensor(np.transpose(img_2, [2, 0, 1]), dtype=torch.float32).unsqueeze(0).cuda()
    print('input.shape:', X1.shape)
    del img_1, img_2, data

    init_lr = args.lr
    model.cuda()
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    Tra_ls = []
    print('trainging begins----------------------------')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        loss = model(X1, X2, idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Tra_ls.append(loss.item())
        if epoch % 10 == 0:
            print('epoch [{}/{}],train:{:.4f}'.format(epoch, args.epochs, loss.item()))
    print('--------------SSL: training is sucessfully down--------------')
    print('--------------model save_path:', args.save_model_path, sep='\n')
    # torch.save(model.state_dict(), args.save_model_path)
    plt.figure(args.EX_num)
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(args.epochs), np.asarray(Tra_ls), 'r-o', label="SSL")
    plt.legend()

    model.eval()
    f1, f2 = model(X1, X2, 0)
    f1, f2 = f1.squeeze(), f2.squeeze()
    f1 = f1.permute(1, 2, 0)
    f2 = f2.permute(1, 2, 0)
    f1 = f1.reshape([-1, C])
    f2 = f2.reshape([-1, C])

    print('input shape:', f1.shape)
    mse_criterion = torch.nn.MSELoss(reduction='none')
    MSE_result= mse_criterion(f1, f2)
    MSE_result = np.mean(MSE_result.numpy(), axis=1).reshape([H, W])

    plt.figure()
    plt.imshow(MSE_result)
    plt.title('MSE')

    # sio.savemat(args.save_result_path, { 'MSE_result': MSE_result})
    print('--------------save_result_path:', args.save_result_path, sep='\n')
    return MSE_result

def diff_RX(img1, img2):
    # img[H, W, C]
    H, W, C = img1.shape
    img1_2d = np.reshape(img1, [H * W, C])
    img2_2d = np.reshape(img2, [H * W, C])
    diff = np.absolute(img1_2d - img2_2d)
    diff_mean = np.mean(diff, axis=0)
    print('diff_mean shape:', diff_mean.shape)  # [1,C]

    diff_cov = np.cov(diff, rowvar=False)
    diff_mean0 = diff - diff_mean  # [H*W,C]
    del img1, img2, C, img1_2d, img2_2d, diff
    T1 = np.matmul(diff_mean0, np.linalg.inv(diff_cov))  # [H*W,C]
    T2 = np.sum(T1 * diff_mean0, axis=1)
    print('shape:', T2.shape)

    T2 = np.reshape(T2, [H, W])
    plt.figure('Diff_RX')
    plt.imshow(T2, cmap='hot')
    return T2


# for evaluating the performance of the anomaly change detection result
def plot_roc(predict, ground_truth):
    """
    INPUTS:
     predict - anomalous change intensity map
     ground_truth - 0or1
    OUTPUTS:
     X, Y for ROC plotting
     auc
    """
    max_value = np.max(ground_truth)
    if max_value != 1:
        ground_truth = ground_truth / max_value

    # initial point（1.0, 1.0）
    x = 1.0
    y = 1.0
    hight_g, width_g = ground_truth.shape
    hight_p, width_p = predict.shape
    if hight_p != hight_g:
        predict = np.transpose(predict)

    ground_truth = ground_truth.reshape(-1)
    predict = predict.reshape(-1)
    # compuate the number of positive and negagtive pixels of the ground_truth
    pos_num = np.sum(ground_truth == 1)
    neg_num = np.sum(ground_truth == 0)
    # step in axis of  X and Y
    x_step = 1.0 / neg_num
    y_step = 1.0 / pos_num
    # ranking the result map
    index = np.argsort(list(predict))
    ground_truth = ground_truth[index]
    """ 
    for i in ground_truth:
     when ground_truth[i] = 1, TP minus 1，one y_step in the y axis, go down
     when ground_truth[i] = 0, FP minus 1，one x_step in the x axis, go left
    """
    X = np.zeros(ground_truth.shape)
    Y = np.zeros(ground_truth.shape)
    for idx in range(0, hight_g * width_g):
        if ground_truth[idx] == 1:
            y = y - y_step
        else:
            x = x - x_step
        X[idx] = x
        Y[idx] = y

    auc = -np.trapz(Y, X)
    if auc < 0.5:
        auc = -np.trapz(X, Y)
        t = X
        X = Y
        Y = t
    print('auc: ', auc)
    return X, Y, auc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def zz(seed):
    print('---------------everything will be ok-------------')
    print('current seed:', seed)


if __name__ == "__main__":

    """ For (ACD)anomalous change detection of the HyperNet """
    # dataset:Viareggio 2013 EX1
    # SEED = np.arange(1, 11)
    # EX1_time = []
    # for i in np.arange(0, 5):
    #     time1 = time.clock()
    #     seed = SEED[i]
    #     print('\n')
    #     args = get_args_viareggio_EX_1(seed)
    #     setup_seed(args.seed)
    #     zz(seed)
    #     train_HyperNet_HACD(args)
    #     time2 = time.clock()
    #     EX1_time.append(time2 - time1)

    # dataset: Viareggio 2013 EX2
    # SEED = np.arange(1, 11)
    # EX1_time = []
    # for i in np.arange(0, 5):
    #     time1 = time.clock()
    #     seed = SEED[i]
    #     print('\n')
    #     args = get_args_viareggio_EX_2(seed)
    #     setup_seed(args.seed)
    #     zz(seed)
    #     train_HyperNet_HACD(args)
    #     time2 = time.clock()
    #     EX1_time.append(time2 -time1)

    # dataset: simulated_Hymap dataset
    # SEED = np.arange(1, 11)
    # EX3_time = []
    # for i in np.arange(0, 5):
    #     time1 = time.clock()
    #     seed = SEED[i]
    #     print('\n')
    #     args = get_args_simulation_EX_3(seed)
    #     setup_seed(args.seed)
    #     zz(seed)
    #     train_HyperNet_HACD(args)
    #     time2 = time.clock()
    #     EX3_time.append(time2 -time1)

    """ For (BCD)binary change detection of the HyperNet """

    # dataset: USA(Hermiston)
    # SEED = np.arange(1, 6)
    # for i in np.arange(0, 1):
    #     seed = SEED[i]
    #     print('\n')
    #     args = get_args_USA(seed)
    #     setup_seed(args.seed)
    #     zz(seed)
    #     train_HyperNet_HBCD(args)

    # dataset: Bay
    # SEED = np.arange(1, 6)
    # for i in np.arange(0, 1):
    #     seed = SEED[i]
    #     print('\n')
    #     args = get_args_Bay(seed)
    #     setup_seed(args.seed)
    #     zz(seed)
    #     train_HyperNet_HBCD(args)

    # dataset: Barbara dataset
    # SEED = np.arange(1, 11)
    # mode = 'Barbara_half1'
    # Barbara_time = []
    # for i in np.arange(0, 5):
    #     time1 = time.clock()
    #     seed = SEED[i]
    #     print('\n')
    #     args = get_args_Barbara(seed, mode)
    #     setup_seed(args.seed)
    #     zz(seed)
    #     train_HyperNet_HBCD(args)
    #     time2 = time.clock()
    #     Barbara_time.append(time2 - time1)










