__author__ = 'jr'


def init_history():
    history = {}
    history["objval"] = list()
    history["r_norm"] = list()
    history["s_norm"] = list()
    history["eps_pri"] = list()
    history["eps_dual"] = list()

    return history