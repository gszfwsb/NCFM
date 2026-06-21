import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug
import copy
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def main(args):

    
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    channel, im_size, num_classes, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size
    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None


    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)
    
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    soft_cri = SoftCrossEntropy

    image_syn_eval = torch.load(args.data_dir)
    label_syn_eval = torch.load(args.label_dir)
    args.lr_net = torch.load(args.lr_dir)

    for model_eval in model_eval_pool:
        accs_test = []
        accs_train = []
        print('Evaluating: '+model_eval)
        for it_eval in range(5):
            network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
            _, acc_train, acc_test = evaluate_synset(0, copy.deepcopy(network), image_syn_eval, label_syn_eval, testloader, args, texture=False, train_criterion=None)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        print("mean: ", acc_test_mean, "std", acc_test_std)


        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='PathMnist', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data_path', help='dataset path')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--parall_eva', type=bool, default=False, help='dataset')

    parser.add_argument('--data_dir', type=str, default='path', help='dataset')
    parser.add_argument('--label_dir', type=str, default='path', help='dataset')
    parser.add_argument('--lr_dir', type=str, default='path', help='dataset')

    args = parser.parse_args()
    main(args)
