Python scripts for alternating direction method of multipliers

This project gives a port/transcript of the MATLAB implementations of the examples in the paper on distributed optimization with the
alternating direction method of multipliers to Python/NumPy/SciPy.

As output, each example script displays the primal residual r_norm_k, the primal feasibility tolerance eps_pri,
the dual residual s_norm, and the dual feasibility tolerance eps_dual. See section 3.3 of the paper for more
details on these quantities. Also included are plots of the objective value and the primal and dual
residual by iteration.

The matlab sources of the scripts, and examples how to use them, can be found at http://www.stanford.edu/~boyd/papers/admm/
The aforementioned paper "Distributed Optimization and Statistical Learning via the Alternating Direction
Method of Multipliers" from Boyd et al., which describes the derivation of the algorithms, can be found at
http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

