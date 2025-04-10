import os
import sbi
import sbibm
import tqdm
import time
from sympy import ordered
import torch
from sbi.neural_nets.net_builders import build_nsf
import torch
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior
import matplotlib.pyplot as plt
import corner

import numpy as np

n_networks = 5
num_epochs = 350
check_every = 10
n_train_data = 500
n_val_data = 1_000
n_samples = 10_000
KL_tol = 1e-2

example_name = "gaussian_mixture" # "two_moons" #  example_name="two_moons")
task = sbibm.get_task(example_name)
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)

# save_path = "/data/jbga2/projects/sbi_acp/data/"
save_path = "data/"

if not os.path.isdir(save_path):
    os.mkdir(save_path)


lrs = np.append(np.geomspace(1e-2, 1e-3, 4), 1e-3)
decreases = [0.995, 0.95, 0.9, 0.8, 0.8]
# [.85, .825, .8, .775, .775]
# #[.5, .5, .75, .9, .9]
n_equals = 1 + len(lrs) - len(np.unique(lrs))


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


def plot_losses(
    train_losses,
    val_losses,
    num_epochs=num_epochs,
    colors=["r", "g", "b", "y", "purple"],
):

    plt.figure()
    for network_id in range(n_networks):
        plt.plot(train_losses[network_id], c=colors[network_id], alpha=0.3)

        rescale = train_losses[network_id].shape[-1] / val_losses[network_id].shape[-1]
        plt.plot(
            rescale * np.linspace(1, num_epochs, num_epochs),
            val_losses[network_id],
            c=colors[network_id],
            label={network_id},
            # marker="o",
        )

    for i in range(0, int(num_epochs / check_every) + 1):
        plt.axvline(i * rescale * check_every, c="grey", linestyle="--", alpha=0.3)

    plt.legend()


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
        theta = prior.sample((1,))
        x = simulator(theta)[0]
        return theta, x


def build_density_estimator(prior=prior, simulator=simulator):
    dummy_data = NPEData(num_samples=64, prior=prior, simulator=simulator)
    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        input_dim=dummy_data.x.shape[-1],
    )
    return density_estimator


def setup_scheduler(optimizer, step_size=20, gamma=0.8):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, )

    return scheduler


def swap_learning_rates(optimizer1, optimizer2):
    """Swap learning rates between two optimizers."""
    lr1 = optimizer1.param_groups[0]["lr"]
    lr2 = optimizer2.param_groups[0]["lr"]

    optimizer1.param_groups[0]["lr"] = lr2
    optimizer2.param_groups[0]["lr"] = lr1


def swap_schedulers(scheduler1, scheduler2, optimizer1, optimizer2):
    """Swap states between two schedulers."""
    state1 = scheduler1.state_dict()
    state2 = scheduler2.state_dict()

    scheduler1.load_state_dict(state2)
    scheduler2.load_state_dict(state1)

    # optimizer1.param_groups[0]["lr"] = (
    #     scheduler1.base_lrs[0] + scheduler2.base_lrs[0]
    # ) / 2
    # optimizer2.param_groups[0]["lr"] = (
    #     scheduler1.base_lrs[0] + scheduler2.base_lrs[0]
    # ) / 2


def swap_probability(val_loss1, val_loss2, lr1, lr2):
    """Compute the swap probability for two learning rates."""

    # factor1 is always positive, we want to swap if the loss for the
    # first network is lower, so for val_loss1 < val_loss2 factor2 is positive too
    factor1 = -(1 / lr1 - 1 / lr2)
    factor2 = -(val_loss1 - val_loss2)

    if lr1 == lr2:
        to_return = min(
            1,
            np.exp(factor2),
        )

    else:
        to_return = min(
            1,
            np.exp(factor1 * factor2),
        )

    print(val_loss1, val_loss2, lr1, lr2)
    print(factor1, factor2, to_return, "\n")

    return to_return


def swap_network_states(network1, network2):
    """
    Swap the states (parameters) of two neural networks.

    Parameters:
    -----------
    network1, network2 : torch.nn.Module
        The neural networks whose states will be swapped.
    """
    state1 = network1.state_dict()
    state2 = network2.state_dict()

    network1.load_state_dict(state2)
    network2.load_state_dict(state1)


def swap_optimizer_states(optimizer1, optimizer2):
    """
    Swap the internal states of two optimizers.

    Parameters:
    -----------
    optimizer1, optimizer2 : torch.optim.Optimizer
        The optimizers whose states will be swapped.
    """
    state1 = optimizer1.state_dict()
    state2 = optimizer2.state_dict()

    optimizer1.load_state_dict(state2)
    optimizer2.load_state_dict(state1)


def reset_optimizer(optimizer, model):
    """
    Reset the state of an optimizer and reinitialize it for a given model.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer to reset.
    model : torch.nn.Module
        The model whose parameters the optimizer will manage.
    """
    optimizer.state = {}  # Clear the optimizer state
    optimizer.param_groups[0]["params"] = model.parameters()  # Reassign parameters


def KLval(posterior_i, posterior_j, tol=KL_tol, n_samples=3000, observation=None):
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


def run_inference(
    n_networks=n_networks, num_epochs=num_epochs, example_name="two_moons"
):
    task = sbibm.get_task(example_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    observation = task.get_observation(num_observation=1)
    reference_samples = task.get_reference_posterior_samples(num_observation=1)
    train_data = NPEData(num_samples=n_train_data, prior=prior, simulator=simulator)
    val_data = NPEData(num_samples=n_val_data, prior=prior, simulator=simulator)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
    density_estimators = [build_density_estimator() for _ in range(n_networks)]

    optimizers = [
        AdamW(density_estimator.parameters(), lr=lrs[i])
        for i, density_estimator in enumerate(density_estimators)
    ]

    schedulers = [
        setup_scheduler(optimizers[i], step_size=check_every, gamma=decreases[i])
        for i in range(len(lrs))
    ]

    train_losses = [[] for _ in range(n_networks)]
    val_losses = [[] for _ in range(n_networks)]
    learning_rates = [[] for _ in range(n_networks)]
    divergence = []
    orders = np.arange(n_networks, dtype=int)
    t0 = time.perf_counter()

    epoch = 0
    dd = 10

    while dd > 0.1 and epoch < num_epochs:
        # for epoch in tqdm.tqdm(range(num_epochs)):
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

                # print(
                #     f"Epoch {epoch + 1}, net_id: {network_id}, train loss: {loss.item()}",
                #     end="\r",
                # )
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
            # scheduler.step(epoch_val_loss[network_id])
            scheduler.step()
            learning_rates[network_id].append(scheduler.get_last_lr())

        if (epoch % check_every == 0 and epoch > 1) or epoch == num_epochs - 1:
            print(f"Now at epoch = {epoch}")  # , end=" ")
            posteriors = []
            n_swaps = 0

            KL_vals = np.zeros((n_equals, n_equals))

            for i in range(n_equals):
                posteriors.append(
                    DirectPosterior(density_estimators[orders[-1 - i]], prior)
                )

            for n_i in range(n_equals):
                for n_j in range(n_i + 1, n_equals):

                    # print("Computing KL between networks", n_i, "and", n_j)
                    KL_vals[n_i, n_j] = KLval(
                        posteriors[n_i], posteriors[n_j], observation=observation
                    )

            divergence.append(KL_vals)
            loss_now = np.array(epoch_val_loss)
            lrs_now = np.array(
                [optimizer.param_groups[0]["lr"] for optimizer in optimizers]
            )

            for i in range(n_networks - 1):
                swap_prob = swap_probability(
                    loss_now[orders[i]],
                    loss_now[orders[i + 1]],
                    lrs_now[orders[i]] / np.min(lrs_now),
                    lrs_now[orders[i + 1]] / np.min(lrs_now),
                )

                # print("Swap probability:", swap_prob)

                if np.random.rand() < swap_prob:
                    n_swaps += 1
                    # print("Swapping learning rates and schedulers")

                    swap_learning_rates(
                        optimizers[orders[i]], optimizers[orders[i + 1]]
                    )
                    swap_schedulers(
                        schedulers[orders[i]],
                        schedulers[orders[i + 1]],
                        optimizers[orders[i]],
                        optimizers[orders[i + 1]],
                    )

                    loss_now[orders[i]], loss_now[orders[i + 1]] = (
                        loss_now[orders[i + 1]],
                        loss_now[orders[i]],
                    )

                    lrs_now[orders[i]], lrs_now[orders[i + 1]] = (
                        lrs_now[orders[i + 1]],
                        lrs_now[orders[i]],
                    )

                    orders[i], orders[i + 1] = (orders[i + 1], orders[i])

            print(orders)
            print(loss_now[orders])
            print(lrs_now[orders], "\n")

            # swap_network_states(
            #     optimizers[orders[i]], optimizers[orders[i + 1]]
            # )

            # swap_optimizer_states(
            #     optimizers[orders[i]], optimizers[orders[i + 1]]
            # )

            dd = np.abs(divergence[-1][0, 1])
            print(
                " n_swaps = %1d, divergence = %.2f, total time = %.2f"
                % (n_swaps, dd, time.perf_counter() - t0)
            )
        epoch += 1

    p_samples = np.zeros((10, n_networks, n_samples, observation.shape[-1]))
    for i in range(10):
        observation = task.get_observation(num_observation=i+1)
        j = 0
        for posterior in posteriors:
            p_samples[i,j] = np.array(
                    posterior.sample((n_samples,), x=observation, show_progress_bars=False)
                )
            j += 1

    np.savez(
        save_path + example_name + f"_{epoch}_{check_every}.npz",
        train=train_losses,
        val=val_losses,
        KL=np.array(divergence),
        post_samples=np.array(p_samples),
    )

    return train_losses, val_losses, divergence


if __name__ == "__main__":
    res = run_inference(num_epochs=num_epochs, example_name=example_name)
