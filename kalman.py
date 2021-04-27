import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython import embed

import ellipse

dt = 1

_A = jnp.array([
    [1, 0, dt, 0],  #
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

_B = jnp.array([
    [0, 0],  #
    [0, 0],
    [dt, 0],
    [0, dt]
])

# Measure position
_Cp = jnp.array([
    [1, 0, 0, 0],  #
    [0, 1, 0, 0],
])

# Measure velocity
_Cv = jnp.array([
    [0, 0, 1, 0],  #
    [0, 0, 0, 1]
])


def measure(x, C, V):
    return C @ x + V


def simulate(x, A, B, u, W):
    return A @ x + B @ u + W


def filter(A,
           B,
           C,
           p0,
           v0,
           mu0,
           sigma0,
           nt=100,
           key=jax.random.PRNGKey(0),
           P=0.95,
           name="default"):

    # Add small amount of perturbation to make Q postive-definite
    Q = jnp.array([
        [1e-8, 0, 0, 0],  #
        [0, 1e-8, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = 9 * jnp.identity(2)

    # Create initial state
    x = jnp.stack((p0, v0)).flatten()

    # Sample noise
    W_key, V_key = jax.random.split(key)
    Ws = jax.random.multivariate_normal(W_key,
                                        mean=jnp.zeros(4, ),
                                        cov=Q,
                                        shape=(nt, ))
    Vs = jax.random.multivariate_normal(V_key,
                                        mean=jnp.zeros(2, ),
                                        cov=R,
                                        shape=(nt, ))
    # Generate input
    ts = jnp.arange(nt) * dt
    us = -2.5 * jnp.array([jnp.cos(0.05 * ts), jnp.sin(0.05 * ts)]).T

    # Track state and measurements
    y = measure(x, C, Vs[0])
    xs = [x]
    ys = [y]

    # Track mean and covariance
    mu = mu0
    sigma = sigma0
    mus = [mu]
    sigmas = [sigma]

    for t in ts[1:]:
        u_ = us[t - 1]  # Use previous input

        # Simulate and measure
        x = simulate(x, A, B, u_, Ws[t])
        # print(f"{t = }, {x = }")
        y = measure(x, C, Vs[t])

        xs.append(x)
        ys.append(y)

        # Filter

        # (1) Predict
        _mu = A @ mu + B @ u_
        _sigma = A @ sigma @ A.T + Q

        # (2) Update

        # Calculate Kalman gain
        K = _sigma @ C.T @ jnp.linalg.inv(C @ _sigma @ C.T + R)
        # Correct estimates (_mu, _sigma)
        mu = _mu + K @ (y - C @ _mu)
        sigma = _sigma - K @ C @ _sigma

        mus.append(mu)
        sigmas.append(sigma)

    # Plot state and measurements
    xs = jnp.array(xs)
    ys = jnp.array(ys)

    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1], label='state trajectory')
    plt.scatter(ys[:, 0], ys[:, 1], label="measurements")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Simulation")
    plt.savefig(f"images/{name}_simulate.png", bbox_inches='tight')
    # plt.show()

    # Plot estimates with position ellipses
    mus = jnp.array(mus)
    sigmas = jnp.array(sigmas)

    plt.figure(figsize=(9, 15))
    plt.plot(xs[:, 0], xs[:, 1], label='state trajectory')
    # plt.scatter(ys[:, 0], ys[:, 1], label="measurements")
    plt.plot(mus[:, 0], mus[:, 1], label="estimate")
    plt.scatter(mus[:, 0], mus[:, 1])

    # Plot error ellipses for position
    for mu, sigma in zip(mus[1:], sigmas[1:]):
        p_mu = mu[:2]
        p_sigma = sigma[:2, :2]  # Extract upper-left block
        # embed()
        _, p_ellipse = ellipse._error_ellipse(p_mu, p_sigma, P)
        plt.plot(p_ellipse[:, 0], p_ellipse[:, 1])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Kalman Filtering with Position Error Ellipses")
    plt.savefig(f"images/{name}_filter_position_ellipse.png",
                bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(9, 15))
    plt.plot(xs[:, 0], xs[:, 1], label='state trajectory')
    # plt.scatter(ys[:, 0], ys[:, 1], label="measurements")
    plt.plot(mus[:, 0], mus[:, 1], label="estimate")
    plt.scatter(mus[:, 0], mus[:, 1])

    # Plot error ellipses for velocity
    for mu, sigma in zip(mus[1:], sigmas[1:]):
        p_mu = mu[:2]
        v_mu = mu[2:]
        v_sigma = sigma[2:, 2:]  # Extract lower-right block
        _, v_ellipse = ellipse._error_ellipse(v_mu, v_sigma, P)

        _v_ellipse = v_ellipse + p_mu
        _v_mu = v_mu + p_mu
        # embed()

        plt.plot([p_mu[0], _v_mu[0]], [p_mu[1], _v_mu[1]])
        plt.plot(_v_ellipse[:, 0], _v_ellipse[:, 1])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Kalman Filtering with Velocity Error Ellipses")
    # plt.show()
    plt.savefig(f"images/{name}_filter_velocity_ellipse.png",
                bbox_inches='tight')

    print(f"Finished filtering {name}")

    # embed()


if __name__ == "__main__":
    p0 = jnp.array([1000., 0])
    v0 = jnp.array([0, 50.])

    mu0 = jnp.array([1500, 100, 0, 55])
    sigma0 = jnp.array([
        [250000, 0, 0, 0],  #
        [0, 250000, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # for i in range(3):
    #     filter(_A,
    #            _B,
    #            _Cp,
    #            p0,
    #            v0,
    #            mu0,
    #            sigma0,
    #            key=jax.random.PRNGKey(i),
    #            nt=20,
    #            name=f"p{i+1}")

    v_mu0 = jnp.array([1000, 0, 0, 50])
    v_sigma0 = jnp.array([
        [1, 0, 0, 0],  #
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    filter(_A,
           _B,
           _Cv,
           p0,
           v0,
           v_mu0,
           v_sigma0,
           key=jax.random.PRNGKey(0),
           nt=100,
           name=f"v1")
