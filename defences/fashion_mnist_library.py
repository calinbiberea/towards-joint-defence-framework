# This is a wrapper for keeping evidence of all the defences implemented for Fashion-MNIST
# Imports all the module paths
import sys

sys.path.append("../../")

import defences.fashion_mnist.standard_training as standard_training_utils
import defences.fashion_mnist.adversarial_training as adversarial_training_utils
import defences.fashion_mnist.dual_adversarial_training as dual_adversarial_training_utils
import defences.fashion_mnist.regularization as regularization_utils
import defences.fashion_mnist.framework_training as framework_training_utils

def standard_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_standard"
):
    return standard_training_utils.standard_training(
      trainSetLoader,
      load_if_available,
      load_path
    )


def adversarial_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_adversarial",
  **kwargs
):
    return adversarial_training_utils.adversarial_training(
      trainSetLoader,
      attack_name,
      attack_function,
      load_if_available,
      load_path,
      **kwargs
    )


def cw_adversarial_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_cw",
  **kwargs
):
    return adversarial_training_utils.cw_adversarial_training(
      trainSetLoader,
      load_if_available,
      load_path,
      **kwargs
    )


def interpolated_adversarial_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_interpolated_adversarial",
  **kwargs
):
    return adversarial_training_utils.interpolated_adversarial_training(
      trainSetLoader,
      attack_name,
      attack_function,
      load_if_available,
      load_path,
      **kwargs
    )


def dual_adversarial_training(
  trainSetLoader,
  attack_function1,
  attack_function2,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_dual",
  **kwargs
):
    return dual_adversarial_training_utils.dual_adversarial_training(
      trainSetLoader,
      attack_function1,
      attack_function2,
      load_if_available,
      load_path,
      **kwargs
    )


def triple_adversarial_training(
  trainSetLoader,
  attack_function1,
  attack_function2,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_triple",
  **kwargs
):
    return dual_adversarial_training_utils.triple_adversarial_training(
      trainSetLoader,
      attack_function1,
      attack_function2,
      load_if_available,
      load_path,
      **kwargs
    )


def jacobian_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_jacobian",
  **kwargs
):
    return regularization_utils.jacobian_training(
      trainSetLoader,
      load_if_available,
      load_path,
      **kwargs
    )


def ALP_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_alp",
  **kwargs
):
    return regularization_utils.ALP_training(
      trainSetLoader,
      attack_name,
      attack_function,
      load_if_available,
      load_path,
      **kwargs
    )


def jacobian_ALP_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../data/fashion_mnist/fashion_mnist_jacobian_alp",
  **kwargs
):
    return regularization_utils.ALP_training(
      trainSetLoader,
      attack_name,
      attack_function,
      load_if_available,
      load_path,
      **kwargs
    )

def framework_training(
    trainSetLoader,
    attack_function1,
    attack_function2,
    load_if_available=False,
    load_path="../data/fashion_mnist/fashion_mnist_framework",
    **kwargs
):
    return framework_training_utils.framework_training(
        trainSetLoader,
        attack_function1,
        attack_function2,
        load_if_available,
        load_path,
        **kwargs
    )