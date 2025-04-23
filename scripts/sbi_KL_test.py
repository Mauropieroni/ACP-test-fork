# Global
import os
from tabnanny import check
import sbibm
import time
import torch
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior
import numpy as np

# Local
import utils as ut


n_networks = 4
def_batch_size = 64
def_shuffle = True
which_dataloader = "resample"  # "fixed"  #  'regenerate' #


num_epochs = 300
check_every = 1
n_train_data = 10_000
n_val_data = 1_000
n_samples = 10_000

example_name = "two_moons"


KL_tol = 1e-3
KL_stop = 0.01
lr = 1e-3
decrease = 0.9

save_path = "data"

if not os.path.isdir(save_path):
    os.mkdir(save_path)


task = sbibm.get_task(example_name)
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)


def run_inference(
    n_networks=n_networks, num_epochs=num_epochs, example_name="two_moons"
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
        AdamW(density_estimator.parameters(), lr=lr)
        for density_estimator in density_estimators
    ]

    schedulers = [
        ut.setup_scheduler(optimizer, step_size=check_every, gamma=decrease)
        for optimizer in optimizers
    ]
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
            # scheduler.step(epoch_val_loss[network_id])
            scheduler.step()
            learning_rates[network_id].append(scheduler.get_last_lr())

        posteriors = []

        KL_vals = np.zeros((n_networks, n_networks))

        for network_id, density_estimator in zip(range(n_networks), density_estimators):
            posteriors.append(DirectPosterior(density_estimator, prior))

        for n_i in range(n_networks):
            for n_j in range(n_i + 1, n_networks):

                # print("Computing KL between networks", n_i, "and", n_j)
                KL_vals[n_i, n_j] = ut.KLval(
                    posteriors[n_i],
                    posteriors[n_j],
                    KL_tol,
                    n_samples,
                    observation=observation,
                )

        divergence.append(KL_vals)

    p_samples = []
    for posterior in posteriors:
        p_samples.append(
            np.array(
                posterior.sample((n_samples,), x=observation, show_progress_bars=False)
            )
        )
    print(np.array(p_samples).shape)

    np.savez(
        save_path + example_name + ".npz",
        train=train_losses,
        val=val_losses,
        KL=divergence,
        post_samples=np.array(p_samples),
    )

    return train_losses, val_losses, divergence


if __name__ == "__main__":
    # res = run_inference(num_epochs=500, example_name="two_moons")
    # res = run_inference(num_epochs=500, example_name="gaussian_mixture")
    res = run_inference(num_epochs=num_epochs, example_name="gaussian_linear")
    # res = run_inference(num_epochs=500, example_name="gaussian_linear_uniform")
