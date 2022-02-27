import numpy as np
from math import pi, cos, sin, sqrt

# *****************************************************************************
def adam(xs_0, f, gf, alpha=0.001, gamma_v=0.9, gamma_s=0.999, eps=1e-8, n_iter=2e4):
    """ Implements the ADAM optimization algorithm.

        Args
            xs_0: intial parameters for f
            f: the function being optimize, can be called as f(xs_0)
            gf: the gradient of f, can be called as gf(xs)

        Returns:
            Optimized arguments for f
    """

    xs = xs_0
    vs = np.zeros(len(xs))
    ss = np.zeros(len(xs))


    k = 1
    while k <= n_iter:

        g = gf(xs)

        vs = gamma_v * vs + (1 - gamma_v) * g
        vs_hat = vs / (1 - gamma_v ** k)

        
        ss = gamma_s * ss + (1 - gamma_s) * (g ** 2)
        ss_hat = ss / (1 - (gamma_s ** k))

        xs = xs - alpha * vs_hat / (eps + np.sqrt(ss_hat))

        k += 1

    return xs
# *****************************************************************************
def rms_prop(xs_0, f, gf, alpha=0.001, gamma_s=0.9, eps=1e-8, n_iter=2e4):
    """ Implements the RMSProp optimization algorithm.

        Args
            xs_0: intial parameters for f
            f: the function being optimize, can be called as f(xs_0)
            gf: the gradient of f, can be called as gf(xs)

        Returns:
            Optimized arguments for f
    """

    xs = xs_0
    ss = np.zeros(len(xs))

    k = 1
    while k <= n_iter:

        g = gf(xs)

        ss = gamma_s * ss + (1 - gamma_s) * (g ** 2)

        xs = xs - alpha * g / (eps + np.sqrt(ss))

        k += 1

    return xs


# *****************************************************************************
def branin(xs):
    """ A function for examples. """
    x, y = xs

    a = (y - (5.1 * (x ** 2)) / (4 * (pi ** 2)) + (5 * x) / pi - 6) ** 2
    b = 10 * (1 - 1 / (8 * pi)) * cos (x)

    return a + b + 10

def grad_branin(xs):
    """ The gradient of the branin function """
    x, y = xs

    a = y - (5.1 * (x ** 2)) / (4 * (pi ** 2)) + (5 * x) / pi - 6

    return np.array([
        2 * (5 / pi - (5.1 * x) / (2 * (pi ** 2))) * (a ** 2) - 10 * (1 - 1 / (8 * pi)) * sin(x), 
        2 * a
    ])

# *****************************************************************************

if __name__ == "__main__":
    
    # part b
    print("\npart b")
    theta_1 = [5, 20]
    theta_n = adam(theta_1, branin, grad_branin)

    print(f"theta N = {theta_n}")

    gf_optimized = grad_branin(theta_n)
    print(f"||grad f|| = {sqrt(np.dot(gf_optimized, gf_optimized))}")

    print(f"f = {branin(theta_n)}")


    
    # part c
    print("\npart c")

    theta_1 = [5, 20]
    theta_n = rms_prop(theta_1, branin, grad_branin)

    print(f"theta N = {theta_n}")

    gf_optimized = grad_branin(theta_n)
    print(f"||grad f|| = {sqrt(np.dot(gf_optimized, gf_optimized))}")

    print(f"f = {branin(theta_n)}")

    
    # part d
    print("\npart d")

    alphas = [4, 2, 1, 0.1]


    for alpha in alphas:

        print(f"alpha = {alpha}")

        theta_1 = [5, 20]
        theta_n = adam(theta_1, branin, grad_branin, alpha=alpha)

        print("ADAM:")
        print(f"theta N = {theta_n}")

        gf_optimized = grad_branin(theta_n)
        print(f"||grad f|| = {sqrt(np.dot(gf_optimized, gf_optimized))}")

        print(f"f = {branin(theta_n)}")

        print()
        print("RMS Prop:")

        theta_1 = [5, 20]
        theta_n = rms_prop(theta_1, branin, grad_branin, alpha=alpha)

        print("ADAM:")
        print(f"theta N = {theta_n}")

        gf_optimized = grad_branin(theta_n)
        print(f"||grad f|| = {sqrt(np.dot(gf_optimized, gf_optimized))}")

        print(f"f = {branin(theta_n)}")
