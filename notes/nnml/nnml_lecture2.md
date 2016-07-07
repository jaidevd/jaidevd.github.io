Perceptron Learning
===================

* If the ouput is correct, leave the weights alone.
* If the output is incorrectly negative, add the input vector to the weight vector
* If the output is incorrectly positive, subtract the input vector from the output vector

Weight Space
-------------

* Weight vectors should lie on the same side of the decision plane as the direction in which the training vector points.

Why the learning works
-----------------------

$d_{a}$ is the current weight vector.
$d_{b}$ is any _feasible_ weight vector.

Learning, then, consists of minimizing $d_{a}^{2} + d_{b}^{2}$, i.e. bringing the current vector as close as possible to the feasible vectors.

There are infinitely many feasible vectors.

So we define "generously feasible" weight vectors. These are vectors that get all the samples right by at least the size of the input.

The squared distance to all generously feasible weights is always decreased by at least the squared length of the input sample, i.e. the update vector.

Informal proof
--------------
* Each time the perceptron makes a mistake, the current weight vector moves closer to the generously feasible region.
* It moves by at least the squared length of the input vector.
* So if the input vectors aren't infinitesimally small, the weight vector will eventually reach this region, if it exists.

Limitations of Perceptrons
-----------------------

* If unlimited handcrafted feature engineering is allowed, a Perceptron can do almost anything. In the long run, you need to _learn_ features.
* Handcrafting features almost never generalizes
* ANN research came to a halt because perceptrons were shown to be limited - _Group Invariance Theorem_
* 