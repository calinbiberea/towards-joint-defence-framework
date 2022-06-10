import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

import torchattacks

# This here actually adds the path
sys.path.append("../../")
import models.lenet as lenet

# For Jacobian Regularization
from jacobian import JacobianReg

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


def framework_training(
    trainSetLoader,
    attack_function1,
    attack_function2,
    load_if_available=False,
    load_path="../data/mnist/mnist_framework",
    **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = lenet.LeNet5().to(device)
    model.train()

    jacobian_reg = JacobianReg()
    if "jac_lambda" in kwargs:
        jacobian_reg_lambda = kwargs["jac_lambda"]
    else:
        jacobian_reg_lambda = 0.005

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Check if using epsilon
        if "epsilon" in kwargs:
            epsilon = kwargs["epsilon"]
        else:
            epsilon = None

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

            # Check if more epochs suplied
            if "epochs" in kwargs:
                epochs = kwargs["epochs"]

            # Define the attack
            attack_function2 = torchattacks.CW(model, c=c, steps=steps)

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches", leave=False)):
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
                    epsilon=epsilon,
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
                        epsilon=epsilon,
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

                # Firstly use dual adversarial training
                loss = (loss1 + loss2) / 2

                # Introduce Jacobian regularization
                jacobian_reg_logits = model(images)
                jacobian_reg_loss = jacobian_reg(images, jacobian_reg_logits)

                # Total loss
                loss = loss + jacobian_reg_lambda * jacobian_reg_loss

                # Gradient descent
                loss.backward()
                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model