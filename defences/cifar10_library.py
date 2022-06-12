# This is a wrapper for keeping evidence of all the defences implemented for CIFAR-10
# Imports all the module paths
import sys
sys.path.append("../../")

import defences.cifar10.regularization as regularization_utils
import defences.cifar10.dual_adversarial_training as dual_adversarial_training_utils
import defences.cifar10.adversarial_training as adversarial_training_utils
import defences.cifar10.standard_training as standard_training_utils
import defences.cifar10.framework_training as framework_training_utils


def standard_training(
    trainSetLoader,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_standard"
):
    return standard_training_utils.standard_training(
        trainSetLoader,
        long_training,
        load_if_available,
        load_path
    )


def adversarial_training(
    trainSetLoader,
    attack_name,
    attack_function,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_adversarial",
    **kwargs
):
    return adversarial_training_utils.adversarial_training(
        trainSetLoader,
        attack_name,
        attack_function,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def cw_adversarial_training(
    trainSetLoader,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_cw",
    **kwargs
):
    return adversarial_training_utils.cw_adversarial_training(
        trainSetLoader,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def interpolated_adversarial_training(
    trainSetLoader,
    attack_name,
    attack_function,
    long_training=True,
    load_if_available=False,
    clip=True,
    verbose=False,
    test=False,
    load_path="../data/cifar10/cifar10_interpolated_adversarial",
    **kwargs
):
    return adversarial_training_utils.interpolated_adversarial_training(
        trainSetLoader,
        attack_name,
        attack_function,
        long_training,
        load_if_available,
        clip,
        verbose,
        test,
        load_path,
        **kwargs
    )


def dual_adversarial_training(
    trainSetLoader,
    attack_function1,
    attack_function2,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_dual",
    **kwargs
):
    return dual_adversarial_training_utils.dual_adversarial_training(
        trainSetLoader,
        attack_function1,
        attack_function2,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def triple_adversarial_training(
    trainSetLoader,
    attack_function1,
    attack_function2,
    attack_function3,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_triple",
    **kwargs
):
    return dual_adversarial_training_utils.triple_adversarial_training(
        trainSetLoader,
        attack_function1,
        attack_function2,
        attack_function3,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def jacobian_training(
    trainSetLoader,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_jacobian",
    **kwargs
):
    return regularization_utils.jacobian_training(
        trainSetLoader,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def ALP_training(
    trainSetLoader,
    attack_name,
    attack_function,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_alp",
    **kwargs
):
    return regularization_utils.ALP_training(
        trainSetLoader,
        attack_name,
        attack_function,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def jacobian_ALP_training(
    trainSetLoader,
    attack_name,
    attack_function,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_jacobian_alp",
    **kwargs
):
    return regularization_utils.jacobian_ALP_training(
        trainSetLoader,
        attack_name,
        attack_function,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )


def framework_training(
    trainSetLoader,
    attack_function1,
    attack_function2,
    long_training=True,
    load_if_available=False,
    load_path="../data/cifar10/cifar10_framework",
    **kwargs
):
    return framework_training_utils.framework_training(
        trainSetLoader,
        attack_function1,
        attack_function2,
        long_training,
        load_if_available,
        load_path,
        **kwargs
    )
