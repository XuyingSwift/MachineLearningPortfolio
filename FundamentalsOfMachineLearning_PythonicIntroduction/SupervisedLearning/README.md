# Supervised Learning
## Feature magic in machine learning
* Under the umbrella of machine learning, inputs and features are highly correlated because the input can be transformed to get more insightful features. 
* Extracting the input's insightful features helps the model learn faster and more accurately. 
* Machine learning algorithms analyze and process data to identify custom features, which are used to train models for classification, prediction and other tasks. 
* This inproves the accuracy and reliability of models, helps distinguish between individuals, objects, and events, and improves understanding of complex systems and processes. 

## Classifier and Thresholds:
### Classifier:
Every machine takes an input, performs its respective function on that input, and produces an output. When this machine is configured/trained to predict a category/class label from a prespecified finite sef of categories, it's called a classifier.
* Classification example: Assume we already have a library of functions as classifiers. 

### Prediction confidence:
is the level of certrainty that a machine learning model has in tis predictions, and it can be expressed through hard or predictions. 
#### Hard prediction:
predicting actual class labels (0 or 1) is called hard prediction. (Classifiers are mathematical functions, and the constraint of discrete values on the output makes the function challenging to be approximated from data.)
### Soft prediction:
is the prediction of class probabilities rather than the actual label values. 
* Determining the best threshold
* Thresholds are problem-specific

## Parametric Model:
Parametric models are functions defined by a fixed set of parameters in machine learning. They are assumed to be able to approximate the underlying pattern of the data. By adjusting the values of the parameters during the training process, these models can learn to fit the data and make accurate predictions on new inputs. f(x1, x2) = w1*x1 + w2*x2 + w3*x1^2 + w4*x2^2 + w0
# Parametric Model Explanation

## Overview

This document provides a brief explanation of the concept of a parametric model in the context of mathematical or computational models.

## Parametric Model Definition

In this context, the symbol `x` represents the input to the model, while `w` denotes a set of parameters that influence the model's behavior. The function `f_w(x)` represents the parametric model itself, where:

- `x` is the input.
- `w` are the parameters that the model uses.
- `f_w(x)` is the output of the model, calculated based on the input `x` and the parameters `w`.

## Parameters in supervised learning
The process of estimating the parameters using data is called training, and the result of traning is an instance of the model class. The set of functions from which the training process searches for the best function is known as the hypothesis space, and each function in the hypothesis space is known as a hypothesis. In training, the goal is to search for the best hypothesis from a given hypothesis space.

# Loss function
Given a model class `f_w` for input `x` and target `y`, a loss function `l(f_w(x), y)` gives a measure of the deviation of the prediction `f_w(x)` from the ground truth `y` on the data point `(x, y)`. The goal is to find the instance of the given class `f_w` that minimizes the total loss for all given data points. `l_w(x,y) = ||f_w(x) - y||^2`

### Parameter Space:
The set of all possible parameters for a model is known as parameter space.

### Hyperparameters

Given the model and the loss function, the goal is to find the parameters `w` to best predict the ground truths `y`. But how do we select the model and the loss function in the first place? All we have is a dataset and nothing else. Different models and loss functions might result in better or worse predictions. So, the choice of models and loss functions is also a parameter, as far as the goal is to get the best prediction. For the training process, all parameters, other than the parameters of model class `f_w`, are called hyperparameters. They’re set before training a model and they don’t learn from data. > Note: Some examples of hyperparameters are weight initialization, model selection, feature selection, and loss function.

# Regression
Regression is a technique that creates a mapping function to approximate the relationship between a real number as a target label and its input features.

In other words, given a dataset `D = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}`, where `x_1` and `y_i` represent input and target, respectively.

## Regression vs. Classification
Regression and classification are supervised machine learning techniques that approximate a function to build the relationship between the target labels and input features available. If the target label has some numerical importance (continuous), then it’s termed regression. In contrast, if the target is a class or a categorical value (discrete), then it’s a classification problem.

## Parametric Models
When the function to be approximated is a linear function of the parameters with real numbers as targets, the function approximation process is known as linear regression.

## Single-feature single-target formulation#
* Features are the independent variables of the approximation function, which are tuned to get the dependent variable (target).
* The intercept `w_0` is also known as bias, denoted as `b`. Therefore, the approximation function can also be denoted as `f_w_b(x) = wx + b`.

## Multi-feature single-target formulation
 As there are multiple input features and only one target, the approximation function used in this scenario falls under the category of multi-feature single-target formulation.
 * `y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 ...`

# Overfitting and Underfitting
## What is overfitting?
* A model with many parameters is more flexible in adapting to complex patterns.
* noise: when the training data has accidental irregualrities, a more flexible model runs after the noise to fit the data exactly and often misses the underlying pattern in the data.
* Overfitting is a modeling error where the model aligns too closely with the training data and might not generalize well to unseen data. 
* This means that the model performs exceptionally well on the training data but unsuitable for other data.

