import jax.numpy as jnp

import ellipse

if __name__ == "__main__":
    mu_x = jnp.array([0, 0])
    sigma_xx = jnp.array(
        [
            #
            [0.7, 0.73],
            [0.73, 1.1],
        ]
    )

    mu_y = jnp.array([0, 0])
    sigma_yy = jnp.array(
        [
            #
            [0.7, 0.19],
            [0.19, 0.16],
        ]
    )

    sigma_xy = jnp.array(
        [
            #
            [0.63, 0.23],
            [0.72, 0.31],
        ]
    )
    sigma_yx = sigma_xy.T
    y = jnp.array([0.27, 0.62])

    # Gaussian estimation equations
    mu_xy = mu_x + sigma_xy @ jnp.linalg.inv(sigma_yy) @ (y - mu_y)
    sigma_xy = sigma_xx - sigma_xy @ jnp.linalg.inv(sigma_yy) @ sigma_yx

    print(f"{mu_xy = }, {sigma_xy = }")

    ellipse.plot_error_ellipse(mu_x, sigma_xx, "prior_ellipse.png")
    ellipse.plot_error_ellipse(mu_xy, sigma_xy, "posterior_ellipse.png")
