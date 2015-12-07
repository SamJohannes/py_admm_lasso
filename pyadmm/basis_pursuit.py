__author__ = 'jr'

import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import utils

import time

def objective(A, b, x):
    obj = npl.norm(x, 1)
    return obj


def shrinkage(a, kappa):
    y = np.maximum(0, a - kappa) - np.maximum(0, - a - kappa)
    return y


def basis_pursuit(A, b, rho, alpha):
    QUIET    = True
    MAX_ITER = 250
    ABSTOL   = 1e-4
    RELTOL   = 1e-2

    history = utils.init_history()
    t_start = time.clock();

    n = A.shape[1]

    x = np.zeros((n, 1), dtype=np.float64)
    z = np.zeros((n, 1), dtype=np.float64)
    u = np.zeros((n, 1), dtype=np.float64)

    if not QUIET:
        print "%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n" % ("iter", "r norm", "eps pri", "s norm", "eps dual", "objective")

    At = np.transpose(A)
    AAt = np.dot(A, At)
    P = np.eye(n, dtype=np.float64) - np.dot(At, npl.solve(AAt, A))

    q = np.dot(At, npl.solve(AAt, b))

    for k in range(0, MAX_ITER):

        # x-update
        x = np.dot(P, (z - u)) + q;

        # z-update with relaxation
        zold = z;
        x_hat = alpha * x + (1 - alpha) * zold;
        z = shrinkage(x_hat + u, 1/rho);

        u = u + (x_hat - z);

        # diagnostics, reporting, termination checks
        objval = objective(A, b, x)
        r_norm = npl.norm(x - z)
        s_norm = npl.norm(-rho * (z - zold))
        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * np.maximum(npl.norm(x), npl.norm(-z))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * npl.norm(rho*u)

        history["objval"].append(objval);
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)

        if not QUIET:
            print "%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n" % (k, r_norm, eps_pri, s_norm, eps_dual, objval);

        if (r_norm < eps_pri and s_norm < eps_dual):
            break;

    if not QUIET:
        print "Elapsed time is %f seconds." % (time.clock() - t_start)

    return (z, history)







