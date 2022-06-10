# Like BIM, published in https://arxiv.org/abs/1607.02533
import torch

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


# Iterative Least Likely Class Attack. Note, this is in the same paper
# as BIM / I-FGSM. In particular, this simply goes in the
# other direction of the gradient sign to maximise the effect.
def ifgsm_attack(
    images,
    labels,
    model,
    loss_function,
    epsilon,
    alpha,
    iterations=0,
    scale=False,
    **kwargs
):
    # Clamp value (i.e. make sure pixels lie in 0-255)
    clamp_max = 255

    # Adding clipping to maintain [0,1] range if that is the scale
    if scale:
        clamp_max = clamp_max / 255

    # The paper gives this formula for managing iterations
    if iterations == 0:
        iterations = int(min(epsilon + 4, 1.25 * epsilon))

    for iteration in range(iterations):
        images.requires_grad = True
        logits = model(images)

        # Ensure model stays unchanged, then calculate loss and gradients
        model.zero_grad()
        loss = loss_function(logits, labels).to(device)
        loss.backward()

        # This is the only change, use - instead of +
        fgsm_images = images - alpha * images.grad.sign()

        # Clipping part of the basic iterative method
        # Use 'eps' instead of epsilon to save space
        # Clip attack using X'' = min{255, X + eps, max{0, X - eps, X'}}
        # which is the same as = min{255, min{X + eps, max{max{0, X - eps}, X'}}}

        # max{0, X - eps}
        step1 = torch.clamp(images - epsilon, min=0)

        # max{max{0, X - eps}, X'}
        step2 = (fgsm_images >= step1).float() * fgsm_images + \
            (step1 > fgsm_images).float() * step1

        # min{X + eps, max{max{0, X - eps}, X'}}
        step3 = (step2 > (images + epsilon)).float() * (images +
                                                        epsilon) + (images + epsilon >= step2).float() * step2

        # min{255, min{X + eps, max{max{0, X - eps}, X'}}}
        images = torch.clamp(step3, min=0, max=clamp_max).detach_()
        images = images.to(device)

    # Return the perturbed images
    return images
