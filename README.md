# MSE246-2018-Project

This project implements and compares a number of different models for predicting loan default and loss at default. We use data from the U.S. SBA 504 loan program, consisting of 150,000 loans issued between 1990 and 2014. We augment our data with several macroeconomic factors, including the Consumer Price Index and yearly S&P 500 returns.

## Data Processing

The __data_processed_final__ folder contains our final processed data, created with __data_processed_hujia.ipynb__. __data_exploration.ipynb__ contains code for preliminary analysis and generating exploratory graphs. These graphs can be found in the __graphs__ folder. 

## Logistic Model

The __logistic model.ipynb__ notebook in __logistic_model__ folder contains the code for tuning and analyzing our logistic model. __logistic_roc.csv__ is the validation ROC curve. 

## Neural Network

The __neural_network__ folder contains our attempts at implementing a binary classification neural network. __NNprocessing.py__ contains neural network-specific preprocessing. __static_net.py__ and __dynamic_net.py__ are first attempts, exploring PyTorch's support for dynamic computational graphs. __default_net.py__ contains our final implementation, which uses batch normalization, dropout, and Adam gradient descent. __nn_eval.py__ analyzes our model parameters and tests its validation performance. Unfortunately, were were unable to implement a full functioning neural network. 

## Hazard Model

The hazard model is in the __data_processed_final__ folder, in the __hazard_lifelines_michelle.ipynb__ notebook. 

## Loss Model

The loss model is in the __loss__ folder, in __loss_model_michelle.ipynb__. __1_and_5_year_loss_michelle.ipynb__ contains the tranche loss simulation code. 