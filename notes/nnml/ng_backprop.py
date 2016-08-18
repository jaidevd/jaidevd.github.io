# coding: utf-8
sample = np.array([[0, 0]])
target = np.array([[, 0]])
target = np.array([[1, 0]])
layers = [2, 3, 2]
w1 = rand(3, 2)
w2 = rand(2, 3)
b1 = rand(3, 1)
b2 = rand(2, 1)
a1 = sample.T
a1
dot(w1, a1)
dot(w1, a1.T)
dot(w1, a1) + b1
z2 = dot(w1, a1) + b1
z2
get_ipython().magic(u'ls ')
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
a2 = sigmoid(z2)
z3 = dot(w2, a2) + b2
z3
a3 = sigmoid(z3)
target = target.T
target
a3
del3 = - (target - a3) * a3 * (1 - a3)
del3
dot(w2, del3)
dot(w2.T, del3)
dot(w2.T, del3) * a2 * (1 - a2)
del2 = _
dot(del3, a2.T)
gradw2 = _
w2
gradb2 = del3
dot(del2, a1.T)
gradw1 = _
w1
gradb1 = del2
del2
b1
w1
w2
w1
b1
w2
b2
