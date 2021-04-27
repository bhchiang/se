import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy


def _sqrt(M):
    # return scipy.linalg.sqrtm(M)
    return jax.scipy.linalg.cholesky(M, lower=True)


def _det(M):
    return jnp.linalg.det(M)


def _eta(sigma):
    n = sigma.shape[0]
    return 1 / jnp.sqrt((2 * jnp.pi)**n * _det(sigma))


def _epsilon(sigma, P):
    return (1 - P) / (2 * _det(sigma)**0.5)


def _C(epsilon, eta):
    return -2 * jnp.log(epsilon / eta)


def plot_error_ellipse(mu, sigma, fname, P=0.95):
    ws, xs = _error_ellipse(mu, sigma, P)
    key = jax.random.PRNGKey(0)
    samples = jax.random.multivariate_normal(key, mu, sigma, (1000, ))
    labels = ["standard normal", f"{P * 100}% error ellipse", "distr samples"]

    plt.figure()
    for _x, _label in zip([ws, xs, samples], labels):
        plt.scatter(_x[:, 0], _x[:, 1], label=_label)
    plt.legend()
    # plt.show()
    plt.savefig(fname, bbox_inches="tight")


@jax.jit
def _error_ellipse(mu, sigma, P):
    # Calculate radius of circle to slice standard normal
    epsilon = _epsilon(sigma, P)
    eta = _eta(sigma)
    C = _C(epsilon, eta)
    # print(f"{epsilon = }, {eta = }, {C = }")

    # Sample points on the circle
    thetas = jnp.linspace(0, 2 * jnp.pi, num=100)

    def _xy(theta):
        x = jnp.sqrt(C) * jnp.cos(theta)
        y = jnp.sqrt(C) * jnp.sin(theta)
        return jnp.array([x, y])

    ws = jax.vmap(_xy)(thetas)
    # print(f"{ws.shape = }")

    # Perform change of variables
    _sigma = _sqrt(sigma)

    def _w2x(w):
        return _sigma @ w + mu

    xs = jax.vmap(_w2x)(ws)
    # print(f"{xs.shape = }")
    return ws, xs


if __name__ == "__main__":
    mu_1 = jnp.array([0, 0])
    sigma_1 = jnp.array([
        #
        [0.7, 0.73],
        [0.73, 1.1],
    ])

    mu_2 = jnp.array([5, 5])
    sigma_2 = jnp.array([
        #
        [1, 0.1],
        [0, 2.3],
    ])

    mu_3 = jnp.array([-1, 2])
    sigma_3 = jnp.array([
        #
        [3, 2.3],
        [2.3, 5],
    ])

    plot_error_ellipse(mu_1, sigma_1, "ellipse_1.png")
    plot_error_ellipse(mu_2, sigma_2, "ellipse_2.png")
    plot_error_ellipse(mu_3, sigma_3, "ellipse_3.png")
