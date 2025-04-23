import sbi
import sbibm
import torch
from torch.optim import AdamW
from sbi.neural_nets.net_builders import build_nsf
from sbi.inference.posteriors import DirectPosterior
import matplotlib.pyplot as plt
import corner

import numpy as np

import utils as ut

n_networks = 4
def_batch_size = 64
def_shuffle = True
which_dataloader = "resample"  # "fixed"  #  'regenerate' #


num_epochs = 10
n_train_data = 10_000
n_val_data = 1_000
n_samples = 10_000

lr = 1e-2

task = sbibm.get_task("two_moons")
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)


def build_density_estimator(prior=prior, simulator=simulator):
    embedding_net = ut.EmbeddingNet(in_features=2)
    dummy_data = ut.NPEData(
        num_samples=def_batch_size,
        prior=prior,
        simulator=simulator,
        which_dataloader=which_dataloader,
    )
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


def run_inference(n_networks=n_networks, num_epochs=num_epochs):
    density_estimators = [build_density_estimator() for _ in range(n_networks)]
    optimizers = [
        AdamW(density_estimator.parameters(), lr=lr)
        for density_estimator in density_estimators
    ]

    schedulers = [setup_scheduler(optimizer) for optimizer in optimizers]
    train_losses = [[] for _ in range(n_networks)]
    val_losses = [[] for _ in range(n_networks)]
    learning_rates = [[] for _ in range(n_networks)]
    post_samples = [[] for _ in range(n_networks)]
    log_prob = [[] for _ in range(n_networks)]

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

        for network_id, density_estimator in zip(range(n_networks), density_estimators):
            posterior = DirectPosterior(density_estimator, prior)
            samples = posterior.sample((n_samples,), x=observation)
            post_samples[network_id].append(samples)
            log_prob[network_id].append(posterior.log_prob(samples, x=observation))

    return train_losses, val_losses, post_samples, log_prob


if __name__ == "__main__":
    run_inference()
