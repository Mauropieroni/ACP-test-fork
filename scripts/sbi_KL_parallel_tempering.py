# Global
import os
from random import shuffle
import sbibm
import time
import torch
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior
import numpy as np

# Local
import utils as ut

n_networks = 5
def_batch_size = 64
def_shuffle = True
which_dataloader = "resample"


num_epochs = 350
check_every = 10
n_train_data = 500
n_val_data = 1_000
n_samples = 10_000


KL_tol = 1e-2
KL_stop = 0.1

example_name = "gaussian_linear"  # "gaussian_mixture"  #  example_name="two_moons")


task = sbibm.get_task(example_name)
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)

# save_path = "/data/jbga2/projects/sbi_acp/data/"
save_path = "data/" + str(n_train_data) + "/"

if not os.path.isdir(save_path):
    os.mkdir(save_path)


lrs = np.append(np.geomspace(1e-2, 1e-3, 4), 1e-3)
decreases = [0.995, 0.95, 0.9, 0.8, 0.8]
# [.85, .825, .8, .775, .775]
# #[.5, .5, .75, .9, .9]
n_equals = 1 + len(lrs) - len(np.unique(lrs))


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


def run_inference(
    n_networks=n_networks,
    n_train_data=n_train_data,
    n_val_data=n_val_data,
    num_epochs=num_epochs,
    example_name="two_moons",
    which_dataloader=which_dataloader,
):
    task = sbibm.get_task(example_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    observation = task.get_observation(num_observation=1)
    reference_samples = task.get_reference_posterior_samples(num_observation=1)
    train_data = ut.NPEData(
        num_samples=n_train_data,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    val_data = ut.NPEData(
        num_samples=n_val_data,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=def_batch_size, shuffle=def_shuffle
    )
    density_estimators = [
        ut.build_density_estimator(
            num_samples=def_batch_size,
            prior=prior,
            simulator=simulator,
            which_dataloader=which_dataloader,
        )
        for _ in range(n_networks)
    ]

    optimizers = [
        AdamW(density_estimator.parameters(), lr=lrs[i])
        for i, density_estimator in enumerate(density_estimators)
    ]

    schedulers = [
        ut.setup_scheduler(optimizers[i], step_size=check_every, gamma=decreases[i])
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

    while dd > KL_stop and epoch < num_epochs:
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
                    KL_vals[n_i, n_j] = ut.KLval(
                        posteriors[n_i],
                        posteriors[n_j],
                        KL_tol,
                        n_samples,
                        observation=observation,
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
        observation = task.get_observation(num_observation=i + 1)
        j = 0
        for posterior in posteriors:
            p_samples[i, j] = np.array(
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
