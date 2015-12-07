__author__ = 'jr'

import unittest
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import pyadmm


class LassoTestCase(unittest.TestCase):

    def plot_history(self, history):
        K = len(history["objval"]);
        h = plt.figure()

        plt.plot(range(K), history["objval"])
        plt.ylabel("f(x^k) + g(z^k)")
        plt.xlabel("iter (k)")

        g = plt.figure();
        plt.subplot(2,1,1)
        plt.semilogy(range(K), np.maximum(1e-8, history["r_norm"]), 'k',
                     range(K), history["eps_pri"], 'k--')
        plt.ylabel('||r||_2')

        plt.subplot(2,1,2);
        plt.semilogy(range(K), np.maximum(1e-8, history["s_norm"]), 'k',
                     range(K), history["eps_dual"], 'k--')
        plt.ylabel('||s||_2')
        plt.xlabel('iter (k)')

        plt.show()

    def test_something(self):
        m = 1500;       # number of examples
        n = 5000;       # number of features

        x0 = npr.randn(n, 1)
        alpha = npr.permutation(range(n))[: np.floor(n * 0.5)]
        x0[alpha] = 0

        A = npr.randn(m,n)
        At = np.transpose(A)
        b = np.dot(A, x0) + np.sqrt(0.001) * npr.randn(m,1);

        lmbd_max = npl.norm(np.dot(At, b), np.inf);
        lmbd = 0.1 * lmbd_max;

        x, history = pyadmm.lasso(A, b, lmbd, 1.0, 1.0);

        self.plot_history(history)


if __name__ == '__main__':
    unittest.main()
