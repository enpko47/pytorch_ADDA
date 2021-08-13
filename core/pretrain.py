import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import load_gpu, save_model


def train_src(src_encoder, classifier, src_loader):
    """ Pre-train source encoder and classifier """

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=list(src_encoder.parameters())+list(classifier.parameters()),
                           lr=params.lr_src_enc,
                           betas=(params.beta1, params.beta2))

    # Train model
    for epoch in range(1, params.epochs_pre + 1):
        src_encoder.train()
        classifier.train()

        num_data = 0
        total_acc = 0.0
        total_loss = 0.0

        for images, labels in src_loader:
            # Load images and labels to GPU
            images = load_gpu(images)
            labels = load_gpu(labels)

            # Predict labels and compute loss
            preds = classifier(src_encoder(images))
            loss = criterion(preds, labels)

            # Optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss and total accuracy
            num_data += len(labels)
            total_acc += (preds.max(1)[1] == labels).sum().item()
            total_loss += loss.item()

        total_acc = total_acc / num_data
        total_loss = total_loss / len(src_loader)

        # Print log information
        print('Epoch [{:3}/{:3}] : loss={:8.4f}, acc={:.4f}'.format(
            epoch, params.epochs_pre, total_loss, total_acc
        ))

        # Test model performence
        if epoch % params.test_step_pre == 0:
            test_src(src_encoder, classifier, src_loader)

        # Save model parameters
        if epoch % params.save_step_pre == 0:
            print('\n#=========================================#\n')
            print('\tSave source model parameters\n')
            save_model(src_encoder, 'src-encoder-{}.pt'.format(epoch))
            save_model(classifier, 'classifier-{}.pt'.format(epoch))
            print('\n#=========================================#\n')

    # Save final model parameters
    print('\n#=========================================#\n')
    print('\tSave source model parameters\n')
    save_model(src_encoder, 'src-encoder-final.pt')
    save_model(classifier, 'classifier-final.pt')
    print('\n#=========================================#\n')

    return src_encoder, classifier


def test_src(src_encoder, classifier, src_loader):
    """ Test source encoder and classifier """

    src_encoder.eval()
    classifier.eval()

    # Setup criterion
    criterion = nn.CrossEntropyLoss()

    num_data = 0
    total_acc = 0.0
    total_loss = 0.0

    # Test model
    with torch.no_grad():
        for images, labels in src_loader:
            # load images and labels to GPU
            images = load_gpu(images)
            labels = load_gpu(labels)

            # Predict labels and compute loss
            preds = classifier(src_encoder(images))
            loss = criterion(preds, labels)

            # Update total loss and total accuracy
            num_data += len(labels)
            total_acc += (preds.max(1)[1] == labels).sum().item()
            total_loss += loss.item()

        total_acc = total_acc / num_data
        total_loss = total_loss / len(src_loader)

    print('\n#=========================================#\n')
    print('\t    Test source model\n')
    print('\t   Loss      = {:8.4f}'.format(total_loss))
    print('\t   Accuracy  = {:8.4f}'.format(total_acc))
    print('\n#=========================================#\n')