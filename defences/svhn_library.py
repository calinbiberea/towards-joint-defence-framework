# Given that for the SVHN dataset only one function is implemented,
# only this file has been provided
import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

import torchattacks

# For Jacobian Regularization
from jacobian import JacobianReg

# This here actually adds the path
sys.path.append("../../")
import models.resnet as resnet
import models.resnet_hidden_layer_features as resnet_hidden_layer_features


# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


# Helps adjust learning rate for better results
def adjust_learning_rate(optimizer, epoch, learning_rate, long_training):
    actual_learning_rate = learning_rate
    if long_training:
        first_update_threshold = 100
        second_update_threshold = 150
    else:
        first_update_threshold = 20
        second_update_threshold = 25

    if epoch >= first_update_threshold:
        actual_learning_rate = 0.01
    if epoch >= second_update_threshold:
        actual_learning_rate = 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = actual_learning_rate

# This method creates a new model and also trains it


def standard_training(
    trainSetLoader,
    long_training=True,
    load_if_available=False,
    load_path="../data/svhn/svhn_standard"
):
    # Number of epochs is decided by training length
    if long_training:
        epochs = 200
    else:
        epochs = 30

    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Training Progress"):
            # Print loss results
            total_epoch_loss = 0

            # Adjust the learning rate
            adjust_learning_rate(
                optimizer, epoch, learning_rate, long_training)

            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Clean the gradients
                optimizer.zero_grad()

                # Predict
                logits = model(images)

                # Calculate loss
                loss = loss_function(logits, labels)

                # Gradient descent
                loss.backward()

                # Add total accumulated loss
                total_epoch_loss += loss.item()

                # Also clip the gradients (ReLU leads to vanishing or
                # exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                optimizer.step()

            print("Loss at epoch {} is {}".format(epoch, total_epoch_loss))

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def framework_training(
    trainSetLoader,
    attack_function1,
    attack_function2,
    long_training=True,
    load_if_available=False,
    load_path="../data/svhn/svhn_framework",
    **kwargs
):
    # Number of epochs is decided by training length
    if long_training:
        epochs = 200
    else:
        epochs = 30

    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet_hidden_layer_features.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()

    jacobian_reg = JacobianReg()
    jacobian_reg_lambda = 0.005

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Check if using epsilon
        if "epsilon1" in kwargs:
            epsilon1 = kwargs["epsilon1"]
        else:
            epsilon1 = None

        # Check if using epsilon
        if "epsilon1" in kwargs:
            epsilon2 = kwargs["epsilon1"]
        else:
            epsilon2 = None

        # Check if using alpha
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        else:
            alpha = None

        # Get iterations
        if "iterations" in kwargs:
            iterations = kwargs["iterations"]
        else:
            iterations = None

        use_cw = False

        # Sanity check to use CW
        if attack_function2 is None:
            use_cw = True

            if "steps" in kwargs:
                steps = kwargs["steps"]
            else:
                steps = 1000

            # Check if more epochs suplied
            if "c" in kwargs:
                c = kwargs["c"]
            else:
                c = 1000

            # Define the attack
            attack_function2 = torchattacks.CW(model, c=c, steps=steps)

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
            # Adjust the learning rate
            adjust_learning_rate(
                optimizer, epoch, learning_rate, long_training)

            # Print loss results
            total_epoch_loss = 0

            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Require gradients for Jacobian regularization
                images.requires_grad = True

                # Run the attack
                model.eval()
                perturbed_images1 = attack_function1(
                    images,
                    labels,
                    model,
                    loss_function,
                    epsilon=epsilon1,
                    alpha=alpha,
                    scale=True,
                    iterations=iterations,
                )

                # Include CW checks to allow smarter uses
                if not use_cw:
                    perturbed_images2 = attack_function2(
                        images,
                        labels,
                        model,
                        loss_function,
                        epsilon=epsilon2,
                        alpha=alpha,
                        scale=True,
                        iterations=iterations,
                    )
                else:
                    perturbed_images2 = attack_function2(
                        images,
                        labels,
                    )
                model.train()

                # Predict and optimise
                optimizer.zero_grad()

                logits1 = model(perturbed_images1)
                logits2 = model(perturbed_images2)
                loss1 = loss_function(logits1, labels)
                loss2 = loss_function(logits2, labels)

                loss = (loss1 + loss2) / 2

                # Introduce Jacobian regularization
                jacobian_reg_loss = jacobian_reg(images, model(images))

                # Total loss
                loss = loss + jacobian_reg_lambda * jacobian_reg_loss

                # Gradient descent
                loss.backward()

                # Add total accumulated loss
                total_epoch_loss += loss.item()

                # Also clip the gradients (ReLU leads to vanishing or
                # exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                optimizer.step()

            print("Loss at epoch {} is {}".format(epoch, total_epoch_loss))

            # Also save the model to avoid having fucked up progress
            torch.save(model, load_path + "_" + str(epoch))

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model
