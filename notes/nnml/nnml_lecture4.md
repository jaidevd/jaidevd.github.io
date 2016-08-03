A relational learning task
-----------------------

* Given a large set of triples that come from some family trees, figure out irregularities
* (x has-mom y) & (y has-husband z) => (x has-father y)


The Softmax Output Function
---------------------------

* Squared error measure drawbacks:
	- If the desired output is 1 and actual output is 1e-9, then almost no gradient for the logistic unit to fix the error
	- If assigning probabilities to mutually exclusive classes, outputs should sum to 1.
* Force outputs to represent a probability distribution across discrete alternatives

$$ y_{i} = \frac{e^{z_{i}}}{\sum_{j\in group}e^{z_{i}}} $$

and

$$ \frac{\partial y_{i}}{\partial z_{i}} = y_{i}(1 - y_{i}) $$

* Cost function: negative log probability of the right answers (The Cross-Entropy)

$$ C = -\sum_{j}t_{j} log y_{j} $$

* The steepness of dC/dy exactly balances the flatnes of dy/dz

Neuro-probabilistic language models
----------------------------------

