import torch
import torch.nn as nn

import os
import params


def get_device():
    """ Get device type """

    device = 'cuda:{}'.format(params.gpu_num) if torch.cuda.is_available() else 'cpu'
    return device


def load_gpu(tensor):
    """ Load tensor to GPU """

    device = get_device()
    tensor = tensor.to(device)
    return tensor


def save_model(model, file_name):
    """ Save model parameters """

    # Make root directory for saving model parameters
    if not os.path.exists(params.model_root):
        os.mkdir(params.model_root)

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(params.model_root, file_name))
    print('  - Save model to {}\n'.format(os.path.join(params.model_root, file_name)))


def set_requires_grad(model, requires_grad=True):
    """ Set requires_grad flag """

    for param in model.parameters():
        param.requires_grad = requires_grad


def init_model(model):
    """ Initialize model """

    # Initialize model weight
    for module in model.modules():
        if type(module) in [nn.Conv2d, nn.Linear]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    # Load model to GPU
    model = load_gpu(model)

    return model


def load_model(model, file_name):
    """ Load model parameters """

    if not os.path.exists(os.path.join(params.model_root, file_name)):
        return model, False

    model.load_state_dict(torch.load(os.path.join(params.model_root, file_name), map_location=get_device()))
    return model, True