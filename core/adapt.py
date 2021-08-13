import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import set_requires_grad, load_gpu, save_model


def adda(src_encoder, tgt_encoder, discriminator, src_loader, tgt_loader):
    """ Adversarial adapt to train target encorder """

    src_encoder.eval()
    set_requires_grad(src_encoder, requires_grad=False)

    # Setup criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optim_enc = optim.Adam(params=tgt_encoder.parameters(),
                           lr=params.lr_tgt_enc,
                           betas=(params.beta1, params.beta2))
    optim_dis = optim.Adam(params=discriminator.parameters(),
                           lr=params.lr_dis,
                           betas=(params.beta1, params.beta2))

    for epoch in range(1, params.epochs_adapt + 1):
        num_data = 0
        total_acc = 0.0
        total_loss_enc = 0.0
        total_loss_dis = 0.0

        for (images_src, _), (images_tgt, _) in zip(src_loader, tgt_loader):
            #===========================#
            #    Train discriminator    #
            #===========================#
            set_requires_grad(tgt_encoder, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)

            # Load images to GPU
            images_src = load_gpu(images_src)
            images_tgt = load_gpu(images_tgt)

            # Create domain labels
            labels_dis = torch.cat([torch.ones(images_src.shape[0]),
                                    torch.zeros(images_tgt.shape[0])], 0)
            labels_dis = load_gpu(labels_dis)

            # Extract features
            features_src = src_encoder(images_src)
            features_tgt = tgt_encoder(images_tgt)
            features_concat = torch.cat([features_src, features_tgt], 0)

            # Predict domain labels and compute loss
            preds = discriminator(features_concat).squeeze()
            loss = criterion(preds, labels_dis)

            # Optimize discriminator
            optim_dis.zero_grad()
            loss.backward()
            optim_dis.step()

            # Update loss and accuracy
            num_data += len(labels_dis)
            total_acc += ((preds > 0).long() == labels_dis.long()).sum().item()
            total_loss_dis += loss.item()


            #===========================#
            #    Train target encoder   #
            #===========================#
            set_requires_grad(tgt_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)

            # Load images to GPU
            images_tgt = load_gpu(images_tgt)

            # Create domain labels
            labels = torch.ones(images_tgt.shape[0])
            labels = load_gpu(labels)

            # Predict domain labels and compute loss
            preds = discriminator(tgt_encoder(images_tgt)).squeeze()
            loss = criterion(preds, labels)

            # Optimize target encoder
            optim_enc.zero_grad()
            loss.backward()
            optim_enc.step()

            # Update loss
            total_loss_enc += loss.item()
        
        loader_len = min(len(src_loader), len(tgt_loader))
        total_acc = total_acc / num_data
        total_loss_dis = total_loss_dis / loader_len
        total_loss_enc = total_loss_enc / loader_len

        # Print log information
        print('Epoch [{:3}/{:3}] : g_loss={:9.4f}, d_loss={:9.4f}, acc={:.4f}'.format(
            epoch, params.epochs_adapt, total_loss_enc, total_loss_dis, total_acc
        ))

        # Save model parameters
        if epoch % params.save_step_adapt == 0:
            print('\n#=========================================#\n')
            print('\tSave target model parameters\n')
            save_model(tgt_encoder, 'tgt-encoder-{}.pt'.format(epoch))
            print('#=========================================#\n')
    
    # Save final model parameters
    print('\n#=========================================#\n')
    print('\tSave target model parameters\n')
    save_model(tgt_encoder, 'tgt-encoder-final.pt')
    print('#=========================================#\n')

    return tgt_encoder