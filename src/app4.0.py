import numpy as np


def fx2(x):
    return 2 * x + 3


x_range = np.linspace(-1, 1, 100)
y_value = [fx2(x) for x in x_range]


def adam_matrix(inits, X, Y, lr=0.01, n_iter=1, beta1=0.9, beta2=0.999, epsilon=1e-6):
    n = len(X)
    a, b = inits
    grad_func = [lambda x, y: -2 * x * (y - (a * x + b)), lambda x, y: -2 * (y - (a * x + b))]
    v = np.array([0, 0])
    s = np.array([0, 0])
    a_list, b_list = [a], [b]
    for _ in range(n_iter):
        t = 1
        for i in range(n):
            x_i, y_i = X[i], Y[i]
            grad = np.array([f(x_i, y_i) for f in grad_func])
            # compute the first moment
            v = beta1 * v + (1 - beta1) * grad
            # compute the second moment
            s = beta2 * s + (1 - beta2) * (grad ** 2)

            # normalisation
            v_norm = v / (1 - np.power(beta1, t))
            s_norm = s / (1 - np.power(beta1, t))
            t += 1

            # update gradient
            grad_norm = lr * v_norm / (np.sqrt(s_norm) + epsilon)
            # update params
            a -= grad_norm[0]
            b -= grad_norm[1]

            a_list.append(a)
            b_list.append(b)
    return a_list, b_list

print(y_value)
print(adam_matrix([3,3], x_range, y_value))