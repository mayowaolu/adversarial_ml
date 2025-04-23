import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.manual_seed(198765)


def mart_loss(logits_clean, logits_adv, labels, lambda_reg=5.0):
    # BCE LOSS = standard cross entropy + margin maximization
    # get the probabilty distribution for the adversarial logits
    probs_adv = F.softmax(logits_adv, dim=1)

    # sort it and get the two highest values/prediction
    tmp1 = torch.argsort(input=probs_adv, dim=1)[:,-2:]
    
    # get the max probability for the incorrect class prediction. basically if the highest probability is for the correct class, get the next higest probability
    labels_new = torch.where(tmp1[:,-1] == labels, input=tmp1[:,-2], other=tmp1[:,-1])

    bce_loss = F.cross_entropy(logits_adv, labels) + F.nll_loss(torch.log(1.0001 - probs_adv + 1e-12), labels_new)


    # KL term
    kl = nn.KLDivLoss(reduction='none')
    probs_clean = F.softmax(logits_clean, dim=1)

    probs_true = torch.gather(probs_clean, dim=1, index=(labels.unsqueeze(1)).long()).squeeze()
    #reg_loss = torch.sum( torch.sum(kl(torch.log(probs_adv + 1e-12), probs_clean), dim=1) * (1.000001 - probs_true)) / labels.size(0)
    reg_loss = torch.mean(torch.sum(kl(torch.log(probs_adv + 1e-12), probs_clean), dim=1) * (1.0000001 - probs_true))

    #DEBUG = torch.allclose(reg_loss, N_reg_loss)
    loss = bce_loss + float(lambda_reg) * reg_loss

    return loss


# from torch.autograd import Variable


# def mart_loss_2(rand_start,model,
#               x_natural,
#               y,
#               optimizer,
#               step_size=0.007,
#               epsilon=0.031,
#               perturb_steps=10,
#               beta=5.0,
#               distance='l_inf'):
#     kl = nn.KLDivLoss(reduction='none')
#     model.eval()
#     batch_size = len(x_natural)
#     # generate adversarial example

#     # rand_start = torch.randn(x_natural.shape).cuda().detach()
#     print("rand_start : \n", rand_start[0, 0, 0,0:32])

#     x_adv = x_natural.detach() + 0.001 * rand_start
#     if distance == 'l_inf':
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_ce = F.cross_entropy(model(x_adv), y)
#             grad = torch.autograd.grad(loss_ce, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     # else:
#     #     x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
#     print("adversarial : \n", x_adv[0, 0, 0,0:32])
#     model.train()

#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()

#     logits = model(x_natural)

#     logits_adv = model(x_adv)

#     adv_probs = F.softmax(logits_adv, dim=1)

#     tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

#     new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

#     loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

#     nat_probs = F.softmax(logits, dim=1)

#     true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

#     loss_robust = (1.0 / batch_size) * torch.sum(
#         torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
#     loss = loss_adv + float(beta) * loss_robust

#     return loss


# def mart_loss_3(rand_start,model,
#               x_natural,
#               y,
#               optimizer,
#               step_size=0.007,
#               epsilon=0.031,
#               perturb_steps=10,
#               beta=5.0,
#               distance='l_inf'):
#     kl = nn.KLDivLoss(reduction='none')
#     model.eval()
#     batch_size = len(x_natural)
#     # generate adversarial example
#     #rand_start = torch.randn(x_natural.shape).cuda().detach()
#     print("rand_start : \n", rand_start[0, 0, 0,0:32])
    
#     x_adv = x_natural.detach() + 0.001 * rand_start.detach()

#     if distance == 'l_inf':
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_ce = F.cross_entropy(model(x_adv), y)
#             grad = torch.autograd.grad(loss_ce, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     # else:
#     #     x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
#     print("adversarial : \n", x_adv[0, 0, 0,0:32])
#     model.train()
#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     # zero gradient
#     optimizer.zero_grad()
#     logits_clean = model(x_natural)

#     logits_adv = model(x_adv)

#     labels = y
#     lambda_reg = beta




#     probs_adv = F.softmax(logits_adv, dim=1)

#     # sort it and get the two highest values/prediction
#     tmp1 = torch.argsort(input=probs_adv, dim=1)[:,-2:]
    
#     # get the max probability for the incorrect class prediction. basically if the highest probability is for the correct class, get the next higest probability
#     labels_new = torch.where(tmp1[:,-1] == labels, input=tmp1[:,-2], other=tmp1[:,-1])

#     bce_loss = F.cross_entropy(logits_adv, labels) + F.nll_loss(torch.log(1.0001 - probs_adv + 1e-12), labels_new)


#     # KL term
#     kl = nn.KLDivLoss(reduction='none')
#     probs_clean = F.softmax(logits_clean, dim=1)

#     probs_true = torch.gather(probs_clean, dim=1, index=(labels.unsqueeze(1)).long()).squeeze()
#     #reg_loss = torch.sum( torch.sum(kl(torch.log(probs_adv + 1e-12), probs_clean), dim=1) * (1.000001 - probs_true)) / labels.size(0)
#     reg_loss = torch.mean(torch.sum(kl(torch.log(probs_adv + 1e-12), probs_clean), dim=1) * (1.0000001 - probs_true))

#     #DEBUG = torch.allclose(reg_loss, N_reg_loss)
#     loss = bce_loss + float(lambda_reg) * reg_loss

#     return loss