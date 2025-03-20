import sbi
import sbibm
import torch
from sbi.neural_nets.net_builders import build_nsf
import torch
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior
import matplotlib.pyplot as plt
import corner

import numpy as np

n_networks = 4
num_epochs = 10
n_train_data = 10_000
n_val_data = 1_000
n_samples = 10_000

task = sbibm.get_task("two_moons")
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)


def get_R(samples):
    """
    Computes the Gelman-Rubin (GR) statistic for convergence assessment. The
    GR statistic is a convergence diagnostic used to assess whether multiple
    Markov chains have converged to the same distribution. Values close to 1
    indicate convergence. For details see
    https://en.wikipedia.org/wiki/Gelman-Rubin_statistic

    Parameters:
    -----------
    samples : numpy.ndarray
        Array containing MCMC samples with dimensions
        (N_steps, N_chains, N_parameters).

    Returns:
    --------
    R : numpy.ndarray
        Array containing the Gelman-Rubin statistics indicating convergence for
        the different parameters. Values close to 1 indicate convergence.

    """

    # Get the shapes
    N_steps, N_chains, N_parameters = samples.shape

    # Chain means
    chain_mean = np.mean(samples, axis=0)

    # Global mean
    global_mean = np.mean(chain_mean, axis=0)

    # Variance between the chain means
    variance_of_means = (
        N_steps
        / (N_chains - 1)
        * np.sum((chain_mean - global_mean[None, :]) ** 2, axis=0)
    )

    # Variance of the individual chain across all chains
    intra_chain_variance = np.std(samples, axis=0, ddof=1) ** 2

    # And its averaged value over the chains
    mean_intra_chain_variance = np.mean(intra_chain_variance, axis=0)

    # First term
    term_1 = (N_steps - 1) / N_steps

    # Second term
    term_2 = variance_of_means / mean_intra_chain_variance / N_steps

    # This is the R (as a vector running on the paramters)
    return term_1 + term_2


def plot_losses(train_losses, val_losses, colors=["r", "g", "b", "y"]):

    plt.figure()
    for network_id in range(n_networks):
        plt.plot(train_losses[network_id], c=colors[network_id], alpha=0.3)
        plt.plot(
            157 * np.linspace(1, num_epochs, num_epochs),
            val_losses[network_id],
            c=colors[network_id],
            marker="o",
        )


class EmbeddingNet(torch.nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2),
        )

    def forward(self, x):
        return self.net(x)


class NPEData(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        prior: torch.distributions.Distribution,
        simulator,
        seed: int = 44,
    ):
        super().__init__()
        self.prior = prior
        self.simulator = simulator

        self.theta = prior.sample((num_samples,))
        self.x = simulator(self.theta)

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index: int):
        return self.theta[index, ...], self.x[index, ...]


def build_density_estimator(prior=prior, simulator=simulator):
    embedding_net = EmbeddingNet(in_features=2)
    dummy_data = NPEData(num_samples=64, prior=prior, simulator=simulator)
    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        input_dim=2,
        embedding_net=embedding_net,
        hidden_features=128,
        num_transforms=3,
        num_bins=8,
        tails="linear",
        tail_bound=5,
        apply_unconditional_transform=False,
    )
    return density_estimator


def setup_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    )
    return scheduler


def KLval(posterior_i, posterior_j, tol=1e-2, n_samples=3000, observation=None):
    KL_update, KL_old = 0.0, 0.0
    n_loops = 0
    while np.abs(KL_update - KL_old) > tol or n_loops == 0:
        n_loops += 1
        samples = posterior_i.sample(
            (n_samples,), x=observation, show_progress_bars=False
        )
        log_ratio = posterior_i.log_prob(samples, x=observation) - posterior_j.log_prob(
            samples, x=observation
        )
        KL_old = KL_update
        KL_update = (log_ratio.mean() + KL_old * (n_loops - 1)) / n_loops
    return KL_update


train_data = NPEData(num_samples=n_train_data, prior=prior, simulator=simulator)
val_data = NPEData(num_samples=n_val_data, prior=prior, simulator=simulator)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)


def run_inference(n_networks=n_networks, num_epochs=num_epochs):
    density_estimators = [build_density_estimator() for _ in range(n_networks)]
    optimizers = [
        AdamW(density_estimator.parameters(), lr=1e-3)
        for density_estimator in density_estimators
    ]

    schedulers = [setup_scheduler(optimizer) for optimizer in optimizers]
    train_losses = [[] for _ in range(n_networks)]
    val_losses = [[] for _ in range(n_networks)]
    learning_rates = [[] for _ in range(n_networks)]
    divergence = []

    for epoch in range(num_epochs):
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

                print(
                    f"Epoch {epoch + 1}, net_id: {network_id}, train loss: {loss.item()}",
                    end="\r",
                )
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
            scheduler.step(epoch_val_loss[network_id])
            learning_rates[network_id].append(scheduler.get_last_lr())

        posteriors = []

        KL_vals = np.zeros((n_networks, n_networks))

        for network_id, density_estimator in zip(range(n_networks), density_estimators):
            posteriors.append(DirectPosterior(density_estimator, prior))

        for n_i in range(n_networks):
            for n_j in range(n_i + 1, n_networks):

                print("Computing KL between networks", n_i, "and", n_j)
                KL_vals[n_i, n_j] = KLval(
                    posteriors[n_i], posteriors[n_j], observation=observation
                )

        divergence.append(KL_vals)

    return train_losses, val_losses, divergence


if __name__ == "__main__":
    run_inference()
