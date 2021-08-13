import torch

from utils import load_gpu


def test(encoder, classifier, loader):
    # Set model mode
    encoder.eval()
    classifier.eval()

    num_data = 0
    total_acc = 0.0

    # Test model
    with torch.no_grad():
        for images, labels in loader:
            # load images and labels to GPU
            images = load_gpu(images)
            labels = load_gpu(labels)

            # Predict labels and compute loss
            preds = classifier(encoder(images))

            # Update total loss and total accuracy
            num_data += len(labels)
            total_acc += (preds.max(1)[1] == labels).sum().item()

        total_acc = total_acc / num_data

    print('   - Accuracy = {:.4f}'.format(total_acc))