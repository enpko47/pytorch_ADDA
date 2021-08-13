from datasets import get_usps, get_mnist
from utils import init_model, load_model
from core import train_src, test_src, adda, test
from models import Encoder, Classifier, Discriminator


# Get source and target data loader
src_loader_train = get_mnist(train=True)
src_loader_test = get_mnist(train=False)
tgt_loader_train = get_usps(train=True)
tgt_loader_test = get_usps(train=False)

# Initialize models
src_encoder = init_model(Encoder())
tgt_encoder = init_model(Encoder())
classifier = init_model(Classifier())
discriminator = init_model(Discriminator())


print('#=========================================#\n')
print('Adversarial Discriminator Domain Adaptation\n')
print('#=========================================#\n')


#===============================#
#          Pre-training         #
#===============================#
print('#=========================================#\n')
print('\t\tPre-training\n')
print('#=========================================#\n')

# Load source model parameters
src_encoder, valid_enc = load_model(src_encoder, 'src-encoder-final.pt')
classifier, valid_cls = load_model(classifier, 'classifier-final.pt')

if not valid_enc or not valid_cls:
    src_encoder, classifier = train_src(src_encoder, classifier, src_loader_train)
print('Done!')

# Test source model performance
test_src(src_encoder, classifier, src_loader_test)


#===============================#
#     Adversarial Adaptation    #
#===============================#
print('#=========================================#\n')
print('\t  Adversarial Adaptation\n')
print('#=========================================#\n')

tgt_encoder, _ = load_model(tgt_encoder, 'src-encoder-final.pt')
tgt_encoder, valid_enc = load_model(tgt_encoder, 'tgt-encoder-final.pt')

if not valid_enc:
    tgt_encoder = adda(src_encoder, tgt_encoder, discriminator, src_loader_train, tgt_loader_train)
print('Done!')


#===============================#
#            Testing            #
#===============================#
print('\n#=========================================#\n')
print('\t\tTesting\n')
print('#=========================================#\n')

print('>> Source only\n')
test(src_encoder, classifier, tgt_loader_test)

print('\n>> ADDA\n')
test(tgt_encoder, classifier, tgt_loader_test)