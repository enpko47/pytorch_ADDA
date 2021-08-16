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


## Result

`MNIST → USPS`
|           |    Source only   |       ADDA       |
| --------- | ---------------- | ---------------- |
|   Paper   | 0.752 &pm; 0.016 | 0.894 &pm; 0.002 |
| This repo | 0.749 &pm; 0.017 | 0.934 &pm; 0.003 |


## Data Distribution

`MNIST → USPS`
![mnist_usps](https://user-images.githubusercontent.com/87518376/129594692-8a66fb40-b0eb-41ef-b0ed-b2da53c8a59d.png)


## Reference

- [https://github.com/corenel/pytorch-adda](https://github.com/corenel/pytorch-adda)