# Global
import os
import sbibm
import time
import torch
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior
import numpy as np

# Local
import utils as ut

n_networks = 3
num_epochs = 300
check_every = 10
update_scheduler_every = 15
def_batch_size = 64
def_shuffle = True
n_train_data = 100
n_val_data = n_train_data
n_samples = 10_000
KL_tol = 1e-3
KL_stop = 0.01

which_dataloader = "resample"  # "fixed"  #  'regenerate'
example_name = "gaussian_mixture"

task = sbibm.get_task(example_name)
prior = task.get_prior_dist()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)
reference_samples = task.get_reference_posterior_samples(num_observation=1)

save_path = "KL_data/" + str(n_train_data) + "/"

lrs = [1e-2 for _ in range(n_networks)]
decreases = [0.9 for _ in range(n_networks)]

combinations = []
for i in range(n_networks):
    for j in range(i + 1, n_networks):
        combinations.append((i, j))

n_combinations = int(n_networks * (n_networks - 1) / 2)


def run_inference(
    n_networks=n_networks,
    n_train_data=n_train_data,
    n_val_data=n_val_data,
    num_epochs=num_epochs,
    example_name="two_moons",
    which_dataloader=which_dataloader,
    save_path=save_path,
):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print("Save path: ", save_path)

    task = sbibm.get_task(example_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()
    observation = task.get_observation(num_observation=1)

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
    divergence = []

    t0 = time.perf_counter()

    epoch = 0
    dd = 10

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
                posteriors.append(DirectPosterior(density_estimators[i], prior))

            for k in range(len(combinations)):
                i, j = combinations[k]
                KL_vals[k] = ut.KLval(
                    posteriors[i],
                    posteriors[j],
                    KL_tol,
                    n_samples,
                    observation=observation,
                )

            divergence.append(KL_vals)
            loss_now = np.array(epoch_val_loss)

            dd = np.abs(np.max(divergence[-1]))

            print(
                " divergence = %.3f, total time = %.2f" % (dd, time.perf_counter() - t0)
            )

            kl_val = "{:.2f}".format(dd)

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
        post_samples=p_samples,
    )

    return train_losses, val_losses, divergence


if __name__ == "__main__":

    n_train_data = [100, 200, 500, 1000, 2000, 5000, 10000]
    n_val_data = [2 * v for v in n_train_data]

    for i in range(len(n_train_data)):
        save_path = "KL_data/" + str(n_train_data[i]) + "/"

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        res = run_inference(
            n_train_data=n_train_data[i],
            n_val_data=n_val_data[i],
            n_networks=n_networks,
            num_epochs=num_epochs,
            example_name=example_name,
            save_path=save_path,
            which_dataloader=which_dataloader,
        )
