import torch
import torch.nn.functional as F
from torch.autograd import Variable
#torch.set_default_device("cuda")

"""
    #adversarial
    "num_steps": 10, # pgd num steps
    "epsilon": 0.031, # 8/255
    "step_size": 0.007, # epsilon/4
"""

def pgd_attack(model, inputs, labels, epsilon=0.031, step_size=0.007, num_steps=10, random_start=True, device="cuda"):
    """
    Performs a Projected Gradient Descent (PGD) attack on a model.
    PGD is an iterative adversarial attack that perturbs the input data to maximize
    the loss, constrained by a maximum L-infinity perturbation of epsilon.
    Args:
        model: The neural network model to attack.
        inputs: Input tensor of shape (batch_size, channels, height, width).
        labels: True labels corresponding to the inputs.
        epsilon: Maximum L-infinity norm of the perturbation (default: 0.031).
        step_size: Step size for each iteration (default: 0.007).
        num_steps: Number of PGD iterations (default: 10).
        random_start: Whether to start with a random perturbation (default: True).
        device: Device to perform the attack on (default: "cuda").
    Returns:
        torch.Tensor: Adversarial examples of the same shape as inputs.
    """
    model.eval()
    # Keep original clean samples
    x_clean = inputs.clone().detach()

    delta = torch.rand_like(inputs, device=device) * 2 * epsilon - epsilon
    delta = torch.clamp(delta, -epsilon, epsilon)
    delta.requires_grad = True

    for step in range(num_steps):
        x_adv = x_clean + delta
        
        # for stable batch norm stats and to disable dropout
        with torch.enable_grad():
            outputs = model(x_adv)
        loss = F.cross_entropy(outputs, labels)

        grad = torch.autograd.grad(loss, [delta])[0]
        delta = delta.detach() + step_size * torch.sign(grad)
        delta = torch.clamp(delta, -epsilon, epsilon)

        # clamp adversarial sample to valid pixel range        
        x_adv_clamped = torch.clamp(x_clean + delta, 0.0, 1.0)
        
        # get accurate delat from clamped x
        delta = x_adv_clamped - x_clean
        delta = delta.detach()
        
        if step < num_steps - 1:
            delta.requires_grad = True
    
    # final adversarial sample
    x_adv = (x_clean + delta).detach()
    return x_adv

    # # Keep original clean samples
    # x_clean = inputs.detach()

    # if random_start:
    #     delta = torch.rand_like(inputs, device=device) * 2 * epsilon - epsilon
    # else:
    #     delta = torch.zeros_like(inputs, device=device)
    
    # delta = torch.clamp(delta, -epsilon, epsilon)
    # delta.requires_grad = True

    # with torch.enable_grad():
    #     for _ in range(num_steps):
    #         x_adv = (x_clean + delta).clamp(0,1)

    #         loss = F.cross_entropy(model(x_adv), labels)
    #         grad = torch.autograd.grad(loss, delta)[0]
    #         delta = (delta + step_size * grad.sign()).clamp(-epsilon, epsilon).detach()
    #         delta.requires_grad_(True)

    # x_adv = (x_clean + delta).clamp(0,1).detach()
    # return x_adv



# def pgd_attack(model, inputs, labels, epsilon=0.031, step_size=0.007, num_steps=10, random_start=True, device="cuda"):
#         model.eval()
#         x_natural = inputs.detach()
#         y = labels.detach()
#         # x_adv = x_natural.detach() + 0.001 * rand_start.detach()
#         x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

#         for _ in range(num_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_ce = F.cross_entropy(model(x_adv), y)
#             grad = torch.autograd.grad(loss_ce, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
#         x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
#         return x_adv