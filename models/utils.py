import torch
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal

def smooth_one_hot(t, classes, smoothing = 0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    t_size = torch.Size((*t.shape, classes))
    with torch.no_grad():
        smoothed_one_hot = torch.empty(size = t_size, device = t.device)
        smoothed_one_hot.fill_(smoothing / (classes - 1))
        smoothed_one_hot.view(-1, classes).scatter_(1, t.flatten().unsqueeze(-1), confidence)
    return smoothed_one_hot

def layer_kld(layer):
    # get weight and bias distributions
    weight_mean = layer.weight_mean
    weight_std = layer.weight_logvar.mul(1/2).exp()
    bias_mean = layer.bias_mean
    bias_std = layer.bias_logvar.mul(1/2).exp()

    q_weight = Normal(weight_mean, weight_std)
    q_bias = Normal(bias_mean, bias_std)

    # all layers have a unit Gaussian prior
    p_weight = Normal(torch.zeros_like(weight_mean), torch.ones_like(weight_std))
    p_bias = Normal(torch.zeros_like(bias_mean), torch.ones_like(bias_std))

    weight_kld = kl_divergence(q_weight, p_weight).sum()
    bias_kld = kl_divergence(q_bias, p_bias).sum()

    return weight_kld + bias_kld
