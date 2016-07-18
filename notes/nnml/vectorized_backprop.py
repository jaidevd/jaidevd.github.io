# coding: utf-8
from backprop import sigmoid
X = array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = array([0, 1, 1, 09])
target = array([0, 1, 1, 0])
w1 = np.random.rand(3, 2)
w2 = np.random.rand(3,)
x_hat = c_[X, [1, 1, 1, 1]]
o1 = sigmoid(np.dot(x_hat, w1))
o1
o1_hat = c_[o1, [1, 1, 1, 1]]
w2
w2 = w2.reshape(-1, 1)
w2
o1_hat
np.dot(o1_hat, w2)
o2 = _
d2 = np.array([[o2[0, 0]]])
d2
d1 = o1
o1
e = o2 - target
e
o2
target
e = o2.ravel() - target
e
e = array([[e[0]]])
e
d2
d1
d1 = diag(o1[0, :])
d21
d1
d1 = d1 * (1- d1)
o2
d2
d2 = d2 * (1 - d2)
d1
d2
e
del2 = np.dot(d2, e)
del2
np.dot(d1, dot(w2, del2))
w2
del2
d1
w2
dot(dot(d1, w2), del2)
d1
w2
dot(d1, w2[:, :2])
w2
dot(d1, w2[:, :1])
d1
w2
dot(d1, w2[0, :1])
w2[0, :1]
w2[0, :2]
w2[0, :3]
w2[:2, 0]
w2[:2, :]
dot(d1, w2[:2, :])
dot(dot(d1, w2[:2, :]), del2)
del2
del2
del1 = dot(dot(d1, w2[:2, :]), del2)
del2
del1
del2_add = - alpha * dot(del2, x_hat)
alpha = 0.3
del2_add = - alpha * dot(del2, x_hat)
del2
x_hat
del2_add = - alpha * dot(del2, x_hat[0, :])
del2_add = - alpha * dot(del2, x_hat[0, :].reshape(-1, 1))
del2_add = - alpha * dot(del2, x_hat[0, :].reshape(1, -1))
del2_add
w2
del2_add.T
o1_hat
o_hat
x_jat
x_hat
get_ipython().magic(u'clear ')
o_hat
o1_hat
get_ipython().magic(u'clear ')
del2
-alpha * dot(del2, o1_hat)
-alpha * dot(del2, o1_hat[0, :])
-alpha * dot(del2, o1_hat[0, :].reshape(1, -1))
delw2_add = _
-alpha * dot(del1, x_hat[0, :].reshape(1, -1))
delw1_add = _
w1
delw2_add.T
delw1_add.T
w2
delw2_add.T
d2
e
d1
o
o2
np.diag(o2)
get_ipython().magic(u'pinfo diag')
np.diag(diag(o2))
np.diag(o2.ravel())
D2 = _
E = o2 - target
E
E = o2.ravel() - target
