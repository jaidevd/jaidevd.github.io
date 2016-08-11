Viewpoint Invariance
------------------

* Use redundant invariant features
* Put a box around the object and use normalized pixels
* Use replicated features with pooling (CNNs)

Invariant Feature Approach
--------------------------

* Extract a large redundant set of features that are invariant undet transformations. Eg, pair of roughly parallel lines with a dot between them

Judicious normalization
---------------------

Put a box around the object and use it as a coordinate frame for a set of normalized pixels.

Convolutional Neural Networks for Handwritten Digit Recognition
------------------------------------------------------

* Based on the idea of replicated features
* If a feature detector is useful in one place in the image, it's likely that it will be useful in another place in the image as well.
* So build copies of the detectors in different places
* Replication reduces number of free parameters to be learned
* Replicated features are easy to learn with backpropagation
* Easy to modify backprop to have any linear constraint between weights
* Compute gradients as usual, modify them so that they satisfy certain constraints
* Eg,

To constrain $w_{1} = w_{2}$,
You need $\Delta w_{1} = \Delta w_{2}$
compute $\frac{\partial E}{\partial w_{1}}$ and $\frac{\partial E}{\partial w_{2}}$

and use $\frac{\partial E}{\partial w_{1}} +\frac{\partial E}{\partial w_{2}}$ for $w_{1}$ and $w_{2}$

What does replicating the feature achieves?
-------------------------------------

* NOT translation invariance
* What replicated features achieves is equivariance, not invariance
* Representation changes by just as much as the image
* Invariant knowledge
* If you want to achieve invariance in activity - pool the outputs or replicated feature detectors
* => Small amount of invariance in each layer by averaging four neighbouring replicated detectors to give a single output to the next level.
* This reduces number of inputs to the next layer. Works slightly better to take maximum of four neighbouring feature detectors


LeNet
-----

* Many hidden layers
* Many maps of replicated units in each layer
* Pooling of outputs of nearby replicated units


Priors and Prejudice
--------------------

We can put in prior knowledge into the task using domain knowledge:
1. Connectivity
2. Weight constraints
3. Neuron activation functions

Alternatively:
Use prior knowledge to create training data

Put your effort into generating synthetic data, then train a large dumb net on a GPU

