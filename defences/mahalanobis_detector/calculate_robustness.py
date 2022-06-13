"""
Originally created on Sun Oct 25 2018 by Kimin Lee.
Modified for the purposes of this repository.
"""
from __future__ import print_function

import regression
import argparse

from sklearn.linear_model import LogisticRegressionCV
from torchvision import transforms

parser = argparse.ArgumentParser(description='Mahalanobis-based detector')
parser.add_argument('--net_type', required=True, help='Model architecture: resnet, lenet')
args = parser.parse_args()
print(args)

def main():
    # Datasets to evaluate on are model dependent
    if args.net_type == 'lenet':
        dataset_list = ['mnist', 'fashion']
    else:
        dataset_list = ['cifar10', 'svhn']

    # The adversarial examples to evaluate against
    adv_test_list = ['FGSM', 'DeepFool', 'CWL2', 'PGD100']

    print('Evaluate the robustness of the Mahalanobis based detector at different magnitudes')
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', \
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    for dataset in dataset_list:
        print('Loading test data for: ', dataset)
        outf = './output/' + args.net_type + '_' + dataset + '/'

        for out in adv_test_list:
            for score in score_list:
                print('Using model trained with attack : ', out, ' and Mahalanobis magnitude ', score)

                # Train the logistic regression
                train_X, train_Y = regression.load_characteristics(score, dataset, out, outf, train=True)
                lr = LogisticRegressionCV(n_jobs=-1).fit(train_X, train_Y)

                # Get the test data against all attacks
                for out2 in adv_test_list:
                    test_X, test_Y = regression.load_characteristics(score, dataset, out2, outf, train=False)

                    results = regression.detection_performance(lr, test_X, test_Y, outf)
                    print("Testing using attack ", out, " against ", out2, " obtaining ", results, "\n")

        print("\n\n\n")
if __name__ == '__main__':
    main()
