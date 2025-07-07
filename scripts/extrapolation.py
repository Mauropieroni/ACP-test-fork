import os
import time
import numpy as np

import torch
from torch.optim import AdamW

import sbi
from sbi.neural_nets.net_builders import build_nsf
from sbi.inference.posteriors import DirectPosterior
import tqdm

import utils as ut

n_freq_samples = 10
obs_std = 0.01

n_networks = 5
def_batch_size = 128
def_shuffle = True

num_epochs = 1000
check_every = 10
update_scheduler_every = 15
n_train_data = 1280
n_val_data = 2000
n_samples = 10_000

KL_tol = 1e-3
KL_stop = 1e-3
lrs = [1e-2 for _ in range(n_networks)]
decreases = [0.9 for _ in range(n_networks)]


which_dataloader = "resample"  # "fixed"  #  'regenerate' #
save_path = "extrapolation_data/" + str(n_train_data) + "/"
uniform_prior = torch.distributions.Uniform(-1, 1)


combinations = []
for i in range(n_networks):
    for j in range(i + 1, n_networks):
        combinations.append((i, j))
        combinations.append((j, i))

n_combinations = int(n_networks * (n_networks - 1) )


def get_estimates(data):
    """
    Get estimates from the data.

    Parameters:
    data (torch.Tensor): Input data.

    Returns:
    torch.Tensor: Estimates.
    """
    return torch.mean(data, dim=0), obs_std / np.sqrt(n_freq_samples)


def generate_gaussian_data(mean, std=obs_std):
    """
    Generate Gaussian data for testing.

    Parameters:
    mean (float): Mean of the Gaussian distribution.
    std (float): Standard deviation of the Gaussian distribution.
    n_samples (int): Number of samples to generate.

    Returns:
    torch.Tensor: Generated Gaussian data.
    """

    return torch.normal(mean, torch.full_like(mean, std))


def simulator(theta, std=obs_std, n_samples=n_freq_samples):
    """
    Simulates data with the correct shape for the density estimator.

    Parameters:
    theta (torch.Tensor): Input parameters.
    std (float): Standard deviation of the Gaussian noise.
    n_samples (int): Number of frequency samples.

    Returns:
    torch.Tensor: Simulated data with shape (batch_size, n_freq_samples).
    """

    mean_to_use = theta.unsqueeze(1).expand(
        -1, n_samples
    )  # Shape: (batch_size, n_samples)
    data = generate_gaussian_data(
        mean_to_use, std=std
    )  # Shape: (batch_size, n_samples)
    return data.squeeze()


def density_estimator_extrapolation(
    num_samples, prior, simulator, which_dataloader=which_dataloader
):
    dummy_data = NNPEData(
        num_samples=num_samples,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )

    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        input_dim=n_freq_samples,
        out_dim=1,
    )

    return density_estimator


class NNPEData(ut.NPEData):
    def _resample_getitem(self, index: int):
        # Resample behavior: resample x for the same theta
        theta = torch.atleast_1d(self.theta[index])  #
        x = self.simulator(theta)
        return theta, x


def my_KLval(posterior_i, posterior_j, tol, n_samples, observation=None):
    KL_update, KL_old = 0.0, 0.0
    n_loops = 0

    while np.abs(KL_update - KL_old) > tol or n_loops == 0:
        n_loops += 1
        samples = posterior_i.sample(
            (n_samples,), x=observation, show_progress_bars=False
        )

        samples = samples.view(n_samples, 1)

        log_ratio = posterior_i.log_prob(samples, x=observation) - posterior_j.log_prob(
            samples, x=observation
        )
        KL_old = KL_update
        KL_update = (log_ratio.mean() + KL_old * (n_loops - 1)) / n_loops

    return KL_update


observation_theta = uniform_prior.sample((1,))
observation_data = simulator(observation_theta)


def run_inference(
    n_networks=n_networks,
    n_train_data=n_train_data,
    n_val_data=n_val_data,
    num_epochs=num_epochs,
    which_dataloader=which_dataloader,
    save_path=save_path,
):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print("Save path: ", save_path, "\n")
    print(f"Will use {n_networks} networks")
    print(f"Will use {n_train_data} training data")
    print(f"Will use {n_val_data} validation data\n")
    
    print("Generating training and validation data...", end=" ")

    train_data = NNPEData(
        num_samples=n_train_data,
        prior=uniform_prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )

    val_data = NNPEData(
        num_samples=n_val_data,
        prior=uniform_prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    print(" done")

    print("Defining dataloaders...", end=" ")
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    print(" done")

    print("Defining density estimators...", end=" ")
    density_estimators = [
        density_estimator_extrapolation(
            num_samples=def_batch_size,
            prior=uniform_prior,
            simulator=simulator,
            which_dataloader=which_dataloader,
        )
        for _ in range(n_networks)
    ]
    print(" done")

    print("Defining optimizers and schedulers...", end=" ")
    optimizers = [
        AdamW(density_estimator.parameters(), lr=lrs[i])
        for i, density_estimator in enumerate(density_estimators)
    ]

    schedulers = [
        ut.setup_scheduler(optimizers[i], step_size=check_every, gamma=decreases[i])
        for i in range(len(lrs))
    ]
    print(" done")

    train_losses = [[] for _ in range(n_networks)]
    val_losses = [[] for _ in range(n_networks)]
    divergence = []

    t0 = time.perf_counter()

    epoch = 0
    dd = 10

    print("\nStarting training...")
    while dd > KL_stop and epoch < num_epochs:

        for density_estimator in density_estimators:
            density_estimator.train()

        for theta, x in train_dataloader:
            network_id = 0
            for density_estimator, optimizer in zip(density_estimators, optimizers):
                loss = density_estimator.loss(theta, x).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses[network_id].append(loss.item())
                network_id += 1

        for density_estimator in density_estimators:
            density_estimator.eval()

        epoch_val_loss = [0.0 for _ in range(n_networks)]

        with torch.no_grad():
            for theta, x in val_dataloader:
                network_id = 0
                for density_estimator in density_estimators:
                    epoch_val_loss[network_id] += (
                        density_estimator.loss(theta, x).mean().item()
                    )
                    network_id += 1

        for network_id, scheduler in zip(range(n_networks), schedulers):
            epoch_val_loss[network_id] /= len(val_dataloader)
            val_losses[network_id].append(epoch_val_loss[network_id])
            scheduler.step()

        if (epoch % check_every == 0 and epoch > 1) or epoch == num_epochs - 1:

            print(f"Now at epoch = {epoch}", end=" ")
            posteriors = []
            KL_vals = np.zeros((n_combinations))

            for i in range(n_networks):
                posteriors.append(DirectPosterior(density_estimators[i], uniform_prior))

            for k in range(len(combinations)):
                i, j = combinations[k]
                KL_vals[k] = my_KLval(
                    posteriors[i],
                    posteriors[j],
                    KL_tol,
                    n_samples,
                    observation=observation_data,
                )

            divergence.append(KL_vals)

            dd = np.abs(np.max(divergence[-1]))

            print("KL divergence values:\n", divergence[-1])
            print(
                " divergence = %.3f, total time = %.2fs\n" % (dd, time.perf_counter() - t0)
            )

            kl_val = "{:.2f}".format(dd)

        epoch += 1

    for i in range(n_networks):
        torch.save(
            density_estimators[i].state_dict(),
            save_path + f"network_{i}_{epoch}_{check_every}.pth",
        )

    p_samples = np.zeros((n_networks, n_samples))
    for i in range(n_networks):
        p_samples[i] = np.array(
            posteriors[i].sample(
                (n_samples,), x=observation_data, show_progress_bars=False
            )
        )

    np.savez(
        save_path + f"_{epoch}_{check_every}.npz",
        train=train_losses,
        val=val_losses,
        KL=np.array(divergence),
        post_samples=p_samples,
    )

    return train_losses, val_losses, divergence


if __name__ == "__main__":

    n_train_data_list = [n_train_data]
    n_val_data = [2 * v for v in n_train_data_list]

    for i in range(len(n_train_data_list)):
        save_path = "new_extrapolation_data/" + str(n_train_data_list[i]) + "/"

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        res = run_inference(
            n_train_data=n_train_data_list[i],
            n_val_data=n_val_data[i],
            n_networks=n_networks,
            num_epochs=num_epochs,
            save_path=save_path,
            which_dataloader=which_dataloader,
        )
