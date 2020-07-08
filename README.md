# trapdoor
Code Implementation for Gotta Catch ’Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks

# Note
The code base currently only support CIFAR dataset and three different attacks (CW, ElasticNet, and PGD). 

More code will be released before the paper is published in Nov 2020. 

# How to train a trapdoored model

There is a pretrained CIFAR trapdoored model in ./model and the trapdoors injected are in ./results

If you would like to train a new model or change the setup: 

`python3 inject_trapdoor.py`


# How to run attack and detection: 

Given a trapdoored model in ./model and pattern stored in ./results. Run: 

`python3 eval_detection.py`

For more 
