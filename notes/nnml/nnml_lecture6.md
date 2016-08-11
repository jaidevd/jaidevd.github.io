Stochastic Gradient Descent - Minibatch gradient descent
----------------------------------------

Problem with gradient descent:

* gradient is large in the direction in which you want to go a small distance
* small in the direction in which you want to go a large distance
* For nonlinear MLPs, the error is locally quadratic, so the same issues are seen

Stochastic Gradient Descent:

* If the dataset is redundant, the gradient obtained on the first half will be almost the same as that obtained on the second half. Therefore it's a waste of time
* Update weights using gradient on the first half and then get a gradient for the new weights on the second half
* Extreme version of this is called online learning.
* Mini-batches need to be balanced for classes!

Basic Minibatch gradient descent:

* guess initial learning rate
* reduce if error worsens
* increase if error falls slowly but consistently
* Towards the end of the mini-batch, reduce the learning rate
* Also turn down the learning rate when the error stops decreasing

Tricks for Mini-batch gradient descent
---------------------------------

Initializing the weights:

* If two hidden weights have the same weights and biases, they will never grow apart. So break the symmetry by allowing small random values
* Make the size of the initial weights proportional to the square root of the fan-in

Shifting the inputs:

* Adding a constant to each component of the input
* transform each component of the input such that averaged over the whole data, the offest is zero
* Isn't this the same as normalizing the data?

Scaling the inputs:

* transform each component so it has unit variance over the whole training set

Decorrelate the inputs:

* works pretty well for a linear neuron
* Apply PCA, remove components with smallest eigenvalues, etc
* Scale remaining components by dividing them with the square roots of their eigenvalues

Common problems in MLPs:

* If the learning rate it big, it drives the neuron to be firmly on or firmly off - so the state depends less on the input and more on the learning rate
* error derivatives will not affect them - they're on a plateau that's mistaken for a local minimum
* in classification if using a squared error or a cross entropy error, the best guessing strategy is to make each output unit always equal to the proportion of time it should be positive
* the network finds this strategy quickly and doesn't make use of the input fast enough
* this is another plateau like a local minimum


Speeding minibatch learning:

* Use momentum: don't use gradient to change the weight, use it to change the velocity - the network remembers previous gradients in the velocity
* separate adaptive learning rate for each parameter - adjust based on empirical measurements (are we making progress? sign of the gradient changes => reduce learning rate, else, increase)
* rmsprop: divide the learning rate for a weight by running average of the magnitudes of recent gradients (minibatch version of just using the sign of the gradient)
* take a fancy method from optimization which makes use of curvature information

The Momentum Method
--------------------

$$ v(t) = \alpha v(t - 1) - \epsilon \frac{\partial E}{\partial w} $$

The effect of the gradient is to increment the previous velocity. The velocity also decays by $\alpha$, which is less than 1

and then
$\Delta w(t) = v(t)$