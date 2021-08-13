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
| This repo | 0.750 &pm; 0.007 | 0.937 &pm; 0.006 |


## Reference

- [https://github.com/corenel/pytorch-adda](https://github.com/corenel/pytorch-adda)