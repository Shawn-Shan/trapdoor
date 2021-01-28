# Gotta Catch 'Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks
### ABOUT

This repository contains code implementation of the paper "[Gotta Catch’Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks](https://www.shawnshan.com/files/publication/trapdoor.pdf)", at *ACM CCS 2020*. The slides are [here](https://www.shawnshan.com/files/publication/trapdoor-ccs-clean.pdf). 
Trapdoor is a defense against adversarial attack developed by researchers at [SANDLab](https://sandlab.cs.uchicago.edu/), University of Chicago.  

### Note
The code base currently only support MNIST dataset (other dataset support coming soon) and two different attacks, CW and PGD. 

More code will be released before the paper is published in Nov 2020. 

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.3.1`
- `numpy==1.16.4`
- `tensorflow-gpu==1.14.1`

Our code is tested on `Python 3.6.8`


### How to train a trapdoored model

There is a pretrained CIFAR trapdoored model in ./model and the trapdoors injected are in ./results

If you would like to train a new model or change the setup: 

`python3 inject_trapdoor.py --dataset mnist`

`python3 inject_trapdoor.py --dataset cifar`




### How to run attack and detection: 

Given a trapdoored model in ./model and pattern stored in ./results. Run: 

`python3 eval_detection.py --dataset mnist --attack pgd`

`python3 eval_detection.py --dataset cifar --attack pgd`

Make sure to change the MODEL_PATH, RES_PATH when running the code on customized models. 

The code will run targeted PGD attack on 3 randomly selected label. It will print out the AUC of detection and the attack success rate at 5% FPR. 

To randomize the neuron matching process as we discussed in Section 7.2:

`python3 eval_detection.py --dataset mnist --filter-ratio 0.1`

### Citation
```
@inproceedings{shan2020gotta,
  title={Gotta catch 'em all: Using honeypots to catch adversarial attacks on neural networks},
  author={Shan, Shawn and Wenger, Emily and Wang, Bolun and Li, Bo and Zheng, Haitao and Zhao, Ben Y},
  journal={Proc. of CCS},
  year={2020}
}
```
