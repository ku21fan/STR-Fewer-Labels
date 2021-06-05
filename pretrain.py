import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from tqdm import tqdm

from utils import Averager, adjust_learning_rate, AttnLabelConverter
from dataset import hierarchical_dataset, AlignCollate_SelfSL, Batch_Balanced_Dataset
from model import Model
from modules.self_supervised import MoCoLoss
from test import validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt, log):
    if opt.self == 'MoCo':
        opt.batch_size = 256

    """ dataset preparation """
    if opt.select_data == 'unlabel':
        select_data = ['U1.Book32', 'U2.TextVQA', 'U3.STVQA']
        batch_ratio = [round(1 / len(select_data), 3)] * len(select_data)

    else:
        select_data = opt.select_data.split('-')
        batch_ratio = opt.batch_ratio.split('-')

    train_loader = Batch_Balanced_Dataset(opt, opt.train_data, select_data, batch_ratio, log, learn_type='self')

    AlignCollate_valid = AlignCollate_SelfSL(opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt, data_type='unlabel')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=False)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')

    """ model configuration """
    if opt.self == 'RotNet':
        model = Model(opt, SelfSL_layer=opt.SelfSL_layer)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)

        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    elif opt.self == 'MoCo':
        model = MoCoLoss(opt, dim=opt.moco_dim, K=opt.moco_k, m=opt.moco_m, T=opt.moco_t)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)
    log.write(repr(model) + '\n')

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)

    # loss averager
    train_loss_avg = Averager()
    valid_loss_avg = Averager()

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f'Trainable params num: {sum(params_num)}')
    log.write(f'Trainable params num: {sum(params_num)}\n')
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr)
    elif opt.self == 'MoCo':
        optimizer = torch.optim.SGD(filtered_parameters, lr=opt.moco_lr,
                                    momentum=opt.moco_SGD_m, weight_decay=opt.moco_wd)
        opt.schedule = opt.moco_schedule
        opt.lr = opt.moco_lr
        opt.lr_drop_rate = opt.moco_lr_drop_rate
    else:
        optimizer = torch.optim.SGD(filtered_parameters, lr=opt.lr,
                                    momentum=opt.momentum, weight_decay=opt.weight_decay)
    print("Optimizer:")
    print(optimizer)
    log.write(repr(optimizer) + '\n')

    if 'super' in opt.schedule:
        if opt.optimizer == 'sgd':
            cycle_momentum = True
        else:
            cycle_momentum = False

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr,
                                                        cycle_momentum=cycle_momentum,
                                                        div_factor=20, final_div_factor=1000,
                                                        total_steps=opt.num_iter)
        print("Scheduler:")
        print(scheduler)
        log.write(repr(scheduler) + '\n')

    """ final options """
    # print(opt)
    opt_log = '------------ Options -------------\n'
    args = vars(opt)
    for k, v in args.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    log.write(opt_log)
    log.close()

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    iteration = start_iter
    best_score = -1

    # training loop
    for iteration in tqdm(range(start_iter + 1, opt.num_iter + 1), total=opt.num_iter, position=0, leave=True):
        # train part
        if opt.self == 'RotNet':
            image, Self_label = train_loader.get_batch()
            image = image.to(device)

            preds = model(image, SelfSL_layer=opt.SelfSL_layer)
            target = torch.LongTensor(Self_label).to(device)

        elif opt.self == 'MoCo':
            q, k = train_loader.get_batch_two_images()
            q = q.to(device)
            k = k.to(device)
            preds, target = model(im_q=q, im_k=k)

        loss = criterion(preds, target)
        train_loss_avg.add(loss)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        if 'super' in opt.schedule:
            scheduler.step()
        else:
            adjust_learning_rate(optimizer, iteration, opt)

        # validation part.
        # To see training progress, we also conduct validation when 'iteration == 1'
        if iteration % opt.val_interval == 0 or iteration == 1:
            # for validation log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    length_of_data = 0
                    infer_time = 0
                    n_correct = 0
                    for i, (image_valid, Self_label_valid) in tqdm(enumerate(valid_loader), total=len(valid_loader), position=1, leave=False):
                        if opt.self == 'RotNet':
                            batch_size = image_valid.size(0)
                            start_infer_time = time.time()
                            preds = model(image_valid.to(device), SelfSL_layer=opt.SelfSL_layer)
                            forward_time = time.time() - start_infer_time
                            target = torch.LongTensor(Self_label_valid).to(device)

                        elif opt.self == 'MoCo':
                            batch_size = image_valid.size(0)
                            q_valid = image_valid.to(device)
                            k_valid = Self_label_valid.to(device)
                            start_infer_time = time.time()
                            preds, target = model(im_q=q_valid, im_k=k_valid)
                            forward_time = time.time() - start_infer_time

                        loss = criterion(preds, target)
                        valid_loss_avg.add(loss)
                        infer_time += forward_time
                        _, preds_index = preds.max(1)
                        n_correct += (preds_index == target).sum().item()
                        length_of_data = length_of_data + batch_size

                    current_score = n_correct / length_of_data * 100

                model.train()

                # keep best score (accuracy) model on valid dataset
                if current_score > best_score:
                    best_score = current_score
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_score.pth')

                # validation log: loss, lr, score, time.
                lr = optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                valid_log = f'\n[{iteration}/{opt.num_iter}] Train loss: {train_loss_avg.val():0.5f}, Valid loss: {valid_loss_avg.val():0.5f}, lr: {lr:0.7f}\n'
                valid_log += f'Best_score: {best_score:0.2f}, Current_score: {current_score:0.2f}, '
                valid_log += f'Infer_time: {infer_time:0.1f}, Elapsed_time: {elapsed_time:0.1f}'
                train_loss_avg.reset()
                valid_loss_avg.reset()

                # show some predicted results
                dashed_line = '-' * 80
                if opt.self == 'RotNet':
                    head = f'GT:0 vs Pred | GT:90 vs Pred | GT:180 vs Pred | GT:270 vs Pred'
                    preds_index = preds_index[:20]
                    gts = Self_label_valid[:20]
                elif opt.self == 'MoCo':
                    head = f'GT:0 vs Pred | GT:0 vs Pred | GT:0 vs Pred | GT:0 vs Pred'
                    preds_index = preds_index[:8]
                    gts = torch.zeros(preds_index.shape[0], dtype=torch.long)

                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for i, (gt, pred) in enumerate(zip(gts, preds_index)):
                    if opt.self == 'RotNet':
                        gt, pred = gt * 90, pred * 90
                    if i % 4 != 3:
                        predicted_result_log += f'{gt} vs {pred} | '
                    else:
                        predicted_result_log += f'{gt} vs {pred} \n'
                predicted_result_log += f'{dashed_line}'
                valid_log = f'{valid_log}\n{predicted_result_log}'
                print(valid_log)
                log.write(valid_log + '\n')

    print(f'finished the experiment: {opt.exp_name}, "CUDA_VISIBLE_DEVICES" was {opt.CUDA_VISIBLE_DEVICES}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='data_CVPR2021/training/unlabel/',
                        help='path to training dataset')
    parser.add_argument('--valid_data', default='data_CVPR2021/validation/',
                        help='path to validation dataset')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=200000, help='number of iterations to train for')
    parser.add_argument('--val_interval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--FT', type=str, default='init', help='whether to do fine-tuning |init|freeze|')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer |sgd|adadelta|adam|')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate. default for RotNet')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD. default for RotNet')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight_decay for SGD. default for RotNet')
    parser.add_argument('--schedule', default=[0.3, 0.6, 0.8], nargs='*',
                        help='learning rate schedule (when to drop lr by lr_drop_rate) default for RotNet')
    parser.add_argument('--lr_drop_rate', type=float, default=0.2,
                        help='lr_drop_rate. default for RotNet')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='unlabel',
                        help='select training data default is `unlabel` which means 11 real labeled datasets')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--CHNJPN', action='store_true',
                        help='For CHNJPN, make the long side of image horizontal (rotate the image for vertical image)')
    parser.add_argument('--Aug', type=str, default='None',
                        help='whether to use augmentation |None|mixup|manifold|cutmix|')
    """ Model Architecture """
    parser.add_argument('--model_name', type=str, required=True, help='CRNN|TRBA')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ Self supervised learning """
    parser.add_argument('--self', type=str, default='RotNet',
                        help='whether to use self-supervised learning |RotNet|MoCo|')
    parser.add_argument('--SelfSL_layer', type=str, default='CNN', help='for SelfSL_layer')
    # moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--moco_lr', default=0.03, type=float, help='SGD lr for moco')
    parser.add_argument('--moco_wd', default=0.0001, type=float, help='SGD weight_decay for moco')
    parser.add_argument('--moco_SGD_m', default=0.9, type=float, help='SGD momentum for moco')
    parser.add_argument('--moco_schedule', default=[0.6, 0.8], type=float, help='SGD momentum for moco')
    parser.add_argument('--moco_lr_drop_rate', type=float, default=0.1, help='moco lr_drop_rate')
    """ exp_name and etc """
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--manual_seed', type=int, default=111, help='for random seed setting')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")

    opt = parser.parse_args()

    opt.gpu_name = '_'.join(torch.cuda.get_device_name().split())

    # Use 'NV' for CRNN, 'NR' or 'TR' for TRBA.
    if opt.model_name[0] == 'N':
        opt.Transformation = 'None'
    elif opt.model_name[0] == 'T':
        opt.Transformation = 'TPS'
    else:
        raise

    if opt.model_name[1] == 'V':
        opt.FeatureExtraction = 'VGG'
    elif opt.model_name[1] == 'R':
        opt.FeatureExtraction = 'ResNet'
    else:
        raise

    opt.SequenceModeling = 'None'
    opt.Prediction = 'None'

    if not opt.exp_name:
        opt.exp_name = f'pretrain-{opt.model_name}-{opt.self}-{opt.SelfSL_layer}-{opt.gpu_name}'
        opt.exp_name += f'-Seed{opt.manual_seed}'

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    log = open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a')
    command_line_input = ' '.join(sys.argv)
    log.write(f'Command line input: {command_line_input}\n')

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True
    opt.gpu_name = '_'.join(torch.cuda.get_device_name().split())
    if sys.platform == 'linux':
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print('We recommend to use 1 GPU, check your GPU number, you would miss CUDA_VISIBLE_DEVICES=0 or typo')
        print('To use multi-gpu setting, remove or comment out these lines')
        sys.exit()

    if sys.platform == 'win32':
        opt.workers = 0

    train(opt, log)
