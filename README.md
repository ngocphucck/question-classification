# RNN-pytorch

## Table of contents
* [Introduction](#introduction)
* [Requirements](#requirements)
* [Set up](#set-up)

## Introduction
Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states.

## Architecture
The architecture of the model:
* Embedding layer (pretrained model)
* RNN
* Softmax Loss

## Requirements
Download the embedding pretrained model at [GoogleNews vectorization](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

All requiring packages are stored in file requirements.txt.

## Set up
```bash
$ pip3 install -r requirements.txt
$ python3 train.py
```
