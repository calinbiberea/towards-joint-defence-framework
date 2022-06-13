"""
Originally created on Sun Oct 25 2018 by Kimin Lee.
Modified for the purposes of this repository.
"""
from __future__ import print_function

import argparse
import os

import torch
import numpy as np

import data_loader
import score_generation

from torchvision import transforms
from torch.autograd import Variable

# Imports all the module paths
import sys
sys.path.append("../../")

parser = argparse.ArgumentParser(description='Mahalanobis-based detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='Batch size for the data loader')
parser.add_argument('--dataset', required=True, help='Dataset (arguments): cifar10, svhn, mnist, fashion')
parser.add_argument('--outf', default='./output/', help='Folder path to output results')
parser.add_argument('--net_type', required=True, help='Model architecture: resnet, lenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2 | PGD100 | PGD')
parser.add_argument('--train', required=True, help='Flag for using training data (true) or test data (false)')
args = parser.parse_args()
print(args)


def main():
    # Set the path to output
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    args.num_classes = 10
    os.makedirs(args.outf, exist_ok=True)


    # Move to GPU
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # Load the model and set the load location for the dataset
    if args.net_type == 'resnet':
        in_transform = transforms.Compose([transforms.ToTensor()])

        if args.dataset == 'cifar10':
            model = torch.load("../../data/cifar10/cifar10_framework")
            dataroot = "../../datasets/CIFAR10"
        else:
            # Since the SVHN module was saved as DataParallel, extract the neural network
            model = torch.load("../../data/svhn/svhn_framework").module
            dataroot = "../../datasets/SVHN"

    elif args.net_type == 'lenet':
        in_transform = transforms.Compose([transforms.ToTensor()])

        if args.dataset == 'mnist':
            model = torch.load("../../data/mnist/mnist_framework")
            dataroot = "../../datasets/"
        else:
            model = torch.load("../../data/fashion_mnist/fashion_mnist_framework")
            dataroot = "../../datasets/"

    model.cuda()
    print('Loaded model: ' + args.net_type)

    # Load the data
    train_loader, _ = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, dataroot)
    if args.train == 'true':
        test_clean_data = torch.load(
            args.outf + 'train_clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
        test_adv_data = torch.load(
            args.outf + 'train_adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
        test_noisy_data = torch.load(
            args.outf + 'train_noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
        test_label = torch.load(args.outf + 'train_label_%s_%s_%s.pth' %
                                (args.net_type, args.dataset, args.adv_type))
        print('Loaded train data for: ', args.dataset)
    else:
        test_clean_data = torch.load(
            args.outf + 'test_clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
        test_adv_data = torch.load(
            args.outf + 'test_adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
        test_noisy_data = torch.load(
            args.outf + 'test_noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
        test_label = torch.load(args.outf + 'test_label_%s_%s_%s.pth' %
                                (args.net_type, args.dataset, args.adv_type))
        print('Loaded test data for: ', args.dataset)

    # Set information about feature extaction
    model.eval()
    # The shape depends on the dataset:
    # MNIST/Fashion-MNIST have 28x28 pixel images
    # CIFAR-10/SVHN have 32x32 pixel images
    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        temp_x = torch.rand(2, 3, 32, 32).cuda()
    else:
        temp_x = torch.rand(2, 1, 28, 28).cuda()

    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('Calculate sample mean and covariance')
    sample_mean, precision = score_generation.sample_estimator(
        model, args.num_classes, feature_list, train_loader)

    print('Calculate Mahalanobis scores for various magnitudes')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('\nNoise: ' + str(magnitude))
        for i in range(num_output):
            M_in \
                = score_generation.get_Mahalanobis_score_adv(model, test_clean_data, test_label,
                                                           args.num_classes, args.outf, args.net_type,
                                                           sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate(
                    (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_out \
                = score_generation.get_Mahalanobis_score_adv(model, test_adv_data, test_label,
                                                           args.num_classes, args.outf, args.net_type,
                                                           sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate(
                    (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_noisy \
                = score_generation.get_Mahalanobis_score_adv(model, test_noisy_data, test_label,
                                                           args.num_classes, args.outf, args.net_type,
                                                           sample_mean, precision, i, magnitude)
            M_noisy = np.asarray(M_noisy, dtype=np.float32)
            if i == 0:
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
            else:
                Mahalanobis_noisy = np.concatenate(
                    (Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
        Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

        Mahalanobis_data, Mahalanobis_labels = score_generation.merge_and_generate_labels(
            Mahalanobis_out, Mahalanobis_pos)
        if args.train == 'true':
            file_name = os.path.join(args.outf, 'train_Mahalanobis_%s_%s_%s.npy' % (
                str(magnitude), args.dataset, args.adv_type))
        else:
            file_name = os.path.join(args.outf, 'test_Mahalanobis_%s_%s_%s.npy' % (
                str(magnitude), args.dataset, args.adv_type))

        Mahalanobis_data = np.concatenate(
            (Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)


if __name__ == '__main__':
    main()
