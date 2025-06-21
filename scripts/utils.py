import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi.neural_nets.net_builders import build_nsf


def_dataloader = "fixed"


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


def KLval(posterior_i, posterior_j, tol, n_samples, observation=None):
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
        which_dataloader=def_dataloader,
        seed: int = 44,
    ):
        super().__init__()
        self.prior = prior
        self.simulator = simulator
        self.theta = prior.sample((num_samples,))
        self.x = simulator(self.theta)

        # Set the behavior of __getitem__ based on which_dataloader
        if which_dataloader == "fixed":
            self._getitem_fn = self._fixed_getitem
        elif which_dataloader == "resample":
            self._getitem_fn = self._resample_getitem
        elif which_dataloader == "regenerate":
            self._getitem_fn = self._regenerate_getitem
        else:
            raise ValueError("Invalid dataloader option.")

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index: int):
        return self._getitem_fn(index)

    def _fixed_getitem(self, index: int):
        # Fixed behavior: return precomputed theta and x
        return self.theta[index], self.x[index]

    def _resample_getitem(self, index: int):
        # Resample behavior: resample x for the same theta
        theta = self.theta[index]
        x = self.simulator(theta)[0]
        return theta, x

    def _regenerate_getitem(self, index: int):
        # Resample behavior: regenerate theta and x
        theta = self.prior.sample((1,))
        x = self.simulator(theta)[0]
        return theta, x


def build_density_estimator(
    num_samples, prior, simulator, which_dataloader=def_dataloader
):
    dummy_data = NPEData(
        num_samples=num_samples,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        input_dim=dummy_data.x.shape[-1],
    )
    return density_estimator


def setup_scheduler(optimizer, step_size, gamma=0.8):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    return scheduler


def plot_losses(
    train_losses,
    val_losses,
    check_every,
    colors=["r", "g", "b", "y", "purple"],
    ax=None,
    plot_legend=True,
):

    if ax is None:
        plt.figure()
        ax = plt.gca()

    n_networks, n_train_losses = train_losses.shape
    _, num_epochs = val_losses.shape
    rescale = int(n_train_losses / num_epochs)
    x_vec = (np.arange(n_train_losses) + 1) / rescale
    xx_vec = np.arange(num_epochs) + 1

    for network_id in range(n_networks):
        ax.plot(
            x_vec,
            train_losses[network_id],
            c=colors[network_id],
            alpha=0.3,
            label=f"Training loss network {network_id +1}",
        )
        ax.plot(
            xx_vec,
            val_losses[network_id],
            c=colors[network_id],
            label=f"Validation loss network {network_id +1}",
            # marker="o",
        )

    for i in range(0, int(num_epochs / check_every) + 1):
        ax.axvline(i * check_every, c="grey", linestyle="--", alpha=0.3)

    if plot_legend:
        ax.legend()  # ncols=2)
