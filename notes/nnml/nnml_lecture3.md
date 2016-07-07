Learning Weights of a Linear Neuron
=====================

* In a perceptron, the weights are getting closer to a good set of weights - to a generously feasible set of weights
* In a linear neuron, the outputs are getting closer to the target outputs

* MLP's should never have been called that!

For MLPs, the "proof" of convergence (or of learning, since there may be no convergence), consists of showing that the output values get closer to target values, instead of weights getting closer to weights.

In perceptrons the outputs may get further away from target outputs, even if the learning is perfect.

* Simplest linear neuron is one with a squared error:

$y = \sum_{i}w_{i}x_{i} = \mathbf{w^{T}x}$

Why don't we just solve it analytically?

* We'd like to understand what neurons are doing, and they might not be solving a symbolic equation.
* We want a method to generalize to mulitple layers

Delta Rule: $ \Delta w_{i} = \epsilon x_{i}(t - y)$

Derivation of Delta Rule
------------------------

Error $$ E = \frac{1}{2}\sum_{n\in Training} (t_{n} - y_{n})^{2} $$

Deriving $E$ w.r.t. weights:

$$ \frac{\partial E}{\partial w_{i}} = -\sum_{n}x_{i}^{n}(t_{n} - y_{n}) $$

Multiply both sides by $-\epsilon$, and,

$$
\Delta w_{i} = -\epsilon\frac{\partial E}{\partial w_{i}} = \sum_{n}\epsilon x_{i}^{n}(t_{n} - y_{n})
$$

Learning can be very slow if two input dimensions are highly correlated. How is this? How can this be verified?

Eg: If we keep getting the same quantities of ketchup, chips and fish, we can't tell what contributes how much to the total bill. In other words, there's no new data to learn from. If they're almost equal, the learning is slow.

* Online delta rule is similar to the perceptron learning algo


Error Surface of a Linear Neuron
===============================

* In perceptrons, you don't need an error surface? A space made up of weights will do?
* For linear neurons, the error surface is always a quadratic bowl.
* Perceptron and online delta, weights move perpendicular to training case constraints. For batch learning on linear neurons they move perpendicular to the contours of the error surface!

Learning the Weights of a Logistic Output Neuron
====================================

Just multiply the activation, or output, by the _logit_

$$ y = \frac{1}{1 + e^{-z}}$$

where $z = b +\sum_{i}x_{i}w_{i}$

Therefore

$$ \frac{\partial z}{\partial w_{i}} = x_{i}$$

and 

$$ \frac{\partial z}{\partial x_{i}} = w_{i}$$

Now,

$$ \frac{dy}{dz} = y(1-y)$$

Therefore

$$ \frac{\partial y}{\partial w_{i}} = \frac{\partial z}{\partial w_{i}} \times \frac{dy}{dz} = x_{i}y(1-y)$$


THE FUCKING BACKPROP ALGORITHM
==============================

* Instead of using activities of hidden units for training, use error derivatives of them
* 