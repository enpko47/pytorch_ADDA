import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


def save_plot(X, y, d, mode, file_name):
    """ Save data distribution plot """

    # Set plot size
    plt.figure(figsize=(10, 10))

    for i in range(len(d)):
        # Get domain color
        color = (1.0, 0.0, 0.0, 1.0) if d[i] == 1 else (0.0, 0.0, 1.0, 1.0)

        plt.text(x=X[i, 0], y=X[i, 1],
                 s=str(y[i]),
                 color=color,
                 fontdict={'weight': 'bold', 'size': 9})
    
    # Set plot options
    plt.xlim(X[:, 0].min() * 1.2, X[:, 0].max() * 1.2)
    plt.ylim(X[:, 1].min() * 1.2, X[:, 1].max() * 1.2)
    plt.xticks([]), plt.yticks([])
    plt.title(mode, fontsize=15)
    plt.tight_layout()

    # Make root directory for saving plots
    if not os.path.exists(params.img_root):
        os.mkdir(params.img_root)
    
    # Save plot
    plt.savefig(os.path.join(params.img_root, '{}.png'.format(file_name)))
    print('\n#=========================================#\n')
    print('\tSave {} Distribution\n'.format(mode))
    print('  - {}.png\n'.format(os.path.join(params.img_root, file_name)))
    print('#=========================================#\n')


def visualize_input(src_loader, tgt_loader, file_name):
    """ Visualize input data distribution that reduced dimension by T-SNE """

    # Extract sample data from source dataset
    list_src_images = torch.Tensor()
    list_src_labels = torch.Tensor()

    for idx, (images, labels) in enumerate(src_loader):
        if idx == 5:
            break

        list_src_images = torch.cat([list_src_images, images], 0)
        list_src_labels = torch.cat([list_src_labels, labels], 0)
    
    list_src_domain = torch.ones(list_src_images.shape[0])
    list_src_images = list_src_images.view(list_src_images.shape[0], -1)

    # Extract sample data from target dataset
    list_tgt_images = torch.Tensor()
    list_tgt_labels = torch.Tensor()

    for idx, (images, labels) in enumerate(tgt_loader):
        if idx == 5:
            break

        list_tgt_images = torch.cat([list_tgt_images, images], 0)
        list_tgt_labels = torch.cat([list_tgt_labels, labels], 0)
    
    list_tgt_domain = torch.ones(list_tgt_images.shape[0])
    list_tgt_images = list_tgt_images.view(list_tgt_images.shape[0], -1)

    # Concatenate source and target data
    list_images_concat = torch.cat([list_src_images, list_tgt_images], 0).numpy()
    list_labels_concat = torch.cat([list_src_labels, list_tgt_labels], 0).int().numpy()
    list_domain_concat = torch.cat([list_src_domain, list_tgt_domain], 0).numpy()

    # Reduce dimension by T-SNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    adda_tsne = tsne.fit_transform(list_images_concat)

    # Save T-SNE result
    save_plot(adda_tsne, list_labels_concat, list_domain_concat, 'Input Data', file_name)


def visualize(encoder, src_loader, tgt_loader, mode, file_name):
    """ Visualize source and target data distribution that reduced demansion by T-SNE """

    # Extract sample data from source dataset
    list_src_images = torch.Tensor()
    list_src_labels = torch.Tensor()
    list_src_domain = torch.Tensor()

    for idx, (images, labels) in enumerate(src_loader):
        if idx == 5:
            break

        list_src_images = torch.cat([list_src_images, images], 0)
        list_src_labels = torch.cat([list_src_labels, labels], 0)

    list_src_domain = torch.ones(list_src_images.shape[0])

    # Extract sample data from target dataset
    list_tgt_images = torch.Tensor()
    list_tgt_labels = torch.Tensor()
    list_tgt_domain = torch.Tensor()

    for idx, (images, labels) in enumerate(tgt_loader):
        if idx == 5:
            break

        list_tgt_images = torch.cat([list_tgt_images, images], 0)
        list_tgt_labels = torch.cat([list_tgt_labels, labels], 0)

    list_tgt_domain = torch.zeros(list_tgt_images.shape[0])

    # Concatenate source and target data
    list_images_concat = torch.cat([list_src_images, list_tgt_images], 0)
    list_labels_concat = torch.cat([list_src_labels, list_tgt_labels], 0).int().numpy()
    list_domain_concat = torch.cat([list_src_domain, list_tgt_domain], 0).numpy()

    # Extract features
    list_images_concat = load_gpu(list_images_concat)
    list_features = encoder(list_images_concat).detach().cpu().numpy()

    # Reduce dimension by T-SNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    adda_tsne = tsne.fit_transform(list_features)

    # Save T-SNE result
    save_plot(adda_tsne, list_labels_concat, list_domain_concat, mode, file_name)