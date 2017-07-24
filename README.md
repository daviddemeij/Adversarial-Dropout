# Adversarial-Dropout
Implementation of "Adversarial Dropout for Supervised and Semi-Supervised Learning" by Sungrae Park, Jun-Keon Park, Su-Jin Shin and Il-Chul Moon <a href="https://arxiv.org/abs/1707.03631">https://arxiv.org/abs/1707.03631</a>

Most of the code is based on <a href="https://github.com/takerum/vat_tf">https://github.com/takerum/vat_tf</a>. I simply added an implementation of Virtual Adversarial Dropout to it.

If you find any bugs or improvements please let me know.

## Usage
(Copied from <a href="https://github.com/takerum/vat_tf">https://github.com/takerum/vat_tf</a>)

### Preparation of dataset for semi-supervised learning
On CIFAR-10

```python cifar10.py --data_dir=./dataset/cifar10/```

### Semi-supervised Learning without augmentation
On CIFAR-10

```python train_semisup.py --dataset=cifar10 --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10/ --num_epochs=500 --epoch_decay_start=460 --method=vad```
