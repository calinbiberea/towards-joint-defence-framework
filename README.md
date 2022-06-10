# Towards a Joint-Defence Framework against Evasion Attacks in Deep Learning

This repository contains the software archive used for conducting the experiments for my final year individual project of my MEng degree at Imperial College London. This README.d file should help you navigate around the repository and get an idea of what my thesis presents. The **mahalanobis_detector/** folder contains code taken from [Adversarial Examples Detection and Analysis with Layer-wise Autoencoders](https://github.com/gmum/adversarial_examples_ae_layers), but has been significantly modified to meet the requirements of this project.

## Table of Contents
- [Project Abstract](#project-abstract)
- [Preliminaries](#preliminaries)
- [Authors](#authors)
- [License](#license)

## Project Abstract

The adversarial machine learning space has grown significantly in recent years, as more defences and attacks centred on evasion attacks have been published. While some defences achieve state-of-the-art robustness, the vast majority of defences are evaluated against adversarial examples with small perturbation sizes in order to remain visually indistinguishable from benign inputs. Because industry machine learning pipelines have become fully automated in order to meet the scale requirements of larger and more complex datasets, the adversaries' previous attack algorithms and perturbation size limitations no longer apply. As a result, the attack surface, or the totality attacks and perturbation sizes against which a model must be robust, expands significantly. As Deep Neural Networks gain popularity in industry, it is critical to ensure that models used in security-critical areas such as malware detection and self-driving cars are resistant to adversaries who use arbitrary attacks and perturbation sizes to cover the attack surface. Existing defences can currently do little against such adversaries.

Although different defences can outperform one another when tested against different attacks and perturbation sizes, little research has been conducted into how to select and combine defences to produce a joint-defence framework that can outperform individual defences by covering a larger portion of the attack surface.
This thesis proposes a generalised procedure for selecting and combining defences for an arbitrary classification task in order to investigate the potential of a joint-defence framework in defending against an adversary whose perturbation size is unbounded. To verify the procedure's effectiveness, a joint-defence framework is constructed for the image classification task and evaluated against more comprehensive adversarial robustness criteria presented in this thesis.

The resulting joint-defence framework has been evaluated on the MNIST, Fashion-MNIST, and CIFAR-10 datasets against some of the most popular attacks in the literature (FGSM, PGD, DeepFool, CW), outperforming existing defences by successfully covering a larger portion of the attack surface with less than 5% accuracy loss, whereas existing defences would be unable to defend against different attacks and arbitrary perturbation sizes or would sacrifice more than 10% accuracy to achieve similar results. Furthermore, the framework's transferability and scalability have been proved by showing that the defence parameters obtained for the CIFAR-10 joint-defence framework can be transferred to the SVHN dataset and achieve similar robustness as the individual defences used in the framework.

Additional contributions were made to successfully construct the joint-defence framework for the image classification task. To understand the limitations of existing experiments, the existing results in the literature for adversarial training, ALP and Jacobian-regularization, PCA-based detection, and Mahalanobis-based detection have been extended with additional experiments. Furthermore, a new adversarial training algorithm, N-Attack adversarial training, has been proposed, which can build an adversarially trained model that is robust to different attacks (FGSM, PGD, DeepFool, CW), whereas previous adversarial training methods could only defend against FGSM and PGD adversaries.

Finally, a new attack procedure is developed in this thesis in order to circumvent the created joint-defence framework. Even in this case, the joint-defence framework outperforms existing defences by achieving over 50\% robustness against all studied attacks and perturbation sizes.

## Preliminaries

This project has been tested unde Ubuntu 20.04 (via WSL 2 on Windows 11) and a Python 3.8 environment. Additionally, the main work of this repository has been structured using [Jupyter Notebooks](https://jupyter.org/) and requires the following packages:

* [Pytorch](http://pytorch.org/): support for both GPU and CPU is provided, but using the GPU is recommended due to long execution times.
* [matplotlib](https://matplotlib.org/): for plotting graphs.
* [jacobian](https://github.com/facebookresearch/jacobian_regularizer): a PyTorch implementation of Jacobian regularization.
* [tqdm](https://github.com/tqdm/tqdm): pretty progress bars.
* [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch): a PyTorch implementation of various adversarial attacks, from which CW and DeepFool are used in this repository.

## Repository Structure

The repository consists of adversarial machine learning evasion attacks and defences for the image classification task.

The experiments have been conducted on four different datasets: MNIST, Fashion-MNIST, CIFAR-10 and SVHN.
Unless you are interested in the implemention, I just recommend you take a look at the **notebooks/** folder, as it contains the experiments written as Jupyter Notebooks.

Structure:
- **attacks/**
    - Contains .py files implementing popular evasion attacks (FGSM, BIM, PGD, CW).

- **data/**
    - Contains pre-trained models.
    - You can find additional models (for all the experiments in the notebooks) using the following links:
        - [MNIST](https://drive.google.com/file/d/1UrRky3J2bUn1Z-yR3pU11Gq8oiqziFcB/view?usp=sharing)
        - [Fashion-MNIST](https://drive.google.com/file/d/1A8g1UYD1NXZBoDFqDfxmcP5jzx0OEiRF/view?usp=sharing)
        - [CIFAR-10]()
        - [SVHN]()

- **defences/**
    - Contains .py files implementing popular defences (adversarial training and regularization methods).

- **models/**
    - Contains .py files with the model architectures used.

- **notebooks/**
    - Contains the experiments

- **scripts/**
    - Contains additional scripts using for the GPU cluster provided by university to accelerate experiments.

- **utils/**
    - Various .py files shared between the different modules

## Authors

This was originally created by myself (Calin Biberea) for conducting the experiments for my final year project.

## License

This repository is licensed under the MIT license.