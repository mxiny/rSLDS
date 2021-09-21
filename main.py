import numpy as np
from models import rSLDS
import utils


# Set hyper-parameters

# Model param
trial_num = 50
trial_len = 250
obs_dim = 4
latent_dim = 2
mode_num = 4

# Optimizor paramhist_data
max_iter = 1000


def main():
    # Load data
    z, x, y = utils.load_data("./data/simulated_gaussian_data.npz")

    # Define rSLDS model
    rslds = rSLDS(K=mode_num, N=obs_dim, M=latent_dim, T=trial_len)

    # # Train model
    # elbos, q = rslds.fit(y, max_iter)

    # # Infer latent states
    # x_infer = rslds.infer(y)

    # # Predict new data
    # new_len = 50
    # hist_data = [z, x, y]
    # z_pred, x_pred, y_pred = rslds.sample(new_len, prefix=hist_data)

    # Sample new data
    z_new, x_new, y_new = rslds.sample(10)
    print(z_new)
    


if __name__ == "__main__":
    main()

