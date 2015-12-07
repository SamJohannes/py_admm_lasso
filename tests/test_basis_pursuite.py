__author__ = 'jr'

import unittest
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import pyadmm

class BasisPursuiteTestCase(unittest.TestCase):

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


    def test_one_time(self):

        n = 300
        m = 100
        A = npr.randn(m, n)

        x_orig = npr.randn(n, 1)
        alpha = npr.permutation(range(n))[: np.floor(n * 0.5)]
        x_orig[alpha] = 0;

        b = np.dot(A, x_orig)

        t_start = time.clock();
        x, history = pyadmm.basis_pursuit(A, b, 1.0, 1.0);
        print "Elapsed time is %f seconds." % (time.clock() - t_start)

        self.plot_history(history)



if __name__ == '__main__':
    unittest.main()
