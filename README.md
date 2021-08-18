# ADDA

PyTorch implementation for [Adversarial Discriminative Domain Adaptation](https://arxiv.org/pdf/1702.05464.pdf).


## Environment

```
python 3.6
pytorch 1.7.1
```


## Instruction

```
python main.py
```


## Results

`MNIST → USPS`
|             |      Paper       |    This repo     |
| ----------- | ---------------- | ---------------- |
| Source Only | 0.752 &pm; 0.016 | 0.762 &pm; 0.012 |
|    ADDA     | 0.894 &pm; 0.002 | 0.937 &pm; 0.003 |


## Experiments

`MNIST → USPS`
#### Source Only
|            | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 |
| ---------- | ------ | ------ | ------ | ------ | ------ |
| Source Acc | 0.993  | 0.992  | 0.992  | 0.993  | 0.991  |
| Target Acc | 0.771  | 0.754  | 0.745  | 0.763  | 0.778  |

#### ADDA
|            | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 |
| ---------- | ------ | ------ | ------ | ------ | ------ |
| Source Acc | 0.901  | 0.879  | 0.876  | 0.942  | 0.951  |
| Target Acc | 0.939  | 0.941  | 0.935  | 0.937  | 0.933  |


## Data Distributions

`MNIST → USPS`
![mnist_usps](https://user-images.githubusercontent.com/87518376/129594692-8a66fb40-b0eb-41ef-b0ed-b2da53c8a59d.png)


## Reference

- [https://github.com/corenel/pytorch-adda](https://github.com/corenel/pytorch-adda)