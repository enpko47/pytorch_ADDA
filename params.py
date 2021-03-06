# Parameters for dataset and data loader
data_root = 'data'
batch_size = 128
image_size = 28

# Parameters for optimizing models
lr_src_enc = 0.0001
lr_tgt_enc = 0.00005
lr_dis = 0.001
beta1 = 0.5
beta2 = 0.9

# Parameters for training models
model_root = 'backup'
epochs_pre = 20
test_step_pre = 10
save_step_pre = 10

epochs_adapt = 500
save_step_adapt = 100

# Parameters for GPU
gpu_num = 0

# Parameters for saving images
img_root = 'img'
img_final = 'mnist_usps'
img_input = 'mnist_usps_input'
img_src_only = 'mnist_usps_src_only'
img_adda = 'mnist_usps_adda'