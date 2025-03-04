import sbi
import torch
from sbi.neural_nets.net_builders import build_nsf
from torch.optim import AdamW
from sbi.inference.posteriors import DirectPosterior, ImportanceSamplingPosterior
import matplotlib.pyplot as plt
import corner
from sbi.utils import BoxUniform
import wandb
import tqdm


def simulator(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1


class EmbeddingNet(torch.nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
        )

    def forward(self, x):
        return self.net(x)


class NPEData(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int, prior: torch.distributions.Distribution, simulator
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


def build_density_estimator(prior, simulator):
    embedding_net = EmbeddingNet(in_features=3)
    dummy_data = NPEData(num_samples=64, prior=prior, simulator=simulator)
    density_estimator = build_nsf(
        batch_x=dummy_data.theta,
        batch_y=dummy_data.x,
        embedding_net=embedding_net,
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


if __name__ == "__main__":
    wandb.init(project="npe4gw")

    prior = BoxUniform(low=-2 * torch.ones(3), high=2 * torch.ones(3))

    train_data = NPEData(num_samples=10_000, prior=prior, simulator=simulator)
    val_data = NPEData(num_samples=1_000, prior=prior, simulator=simulator)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)

    density_estimator = build_density_estimator(prior, simulator)
    optimizer = AdamW(density_estimator.parameters(), lr=1e-3)
    scheduler = setup_scheduler(optimizer)
    train_losses = []
    val_losses = []
    learning_rates = []
    post_samples = []

    num_epochs = 100
    step = 0
    epoch_val_loss = 0.0

    for epoch in range(num_epochs):
        density_estimator.train()

        train_loss_epoch = 0.0
        num_batches = len(train_dataloader)
        with tqdm.tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ) as pbar:
            for theta, x in pbar:
                loss = density_estimator.loss(theta, x).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_loss_epoch += loss.item()
                wandb.log({"train_loss": loss.item(), "step": step})
                step += 1
                pbar.set_postfix(
                    {
                        "Train Loss": f"{loss.item():.4f} | Val Loss: {epoch_val_loss:.4f}"
                    }
                )

        avg_train_loss = train_loss_epoch / num_batches

        density_estimator.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for theta, x in val_dataloader:
                epoch_val_loss += density_estimator.loss(theta, x).mean().item()

        epoch_val_loss /= len(val_dataloader)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        learning_rates.append(scheduler.get_last_lr())
        wandb.log(
            {
                "val_loss": epoch_val_loss,
                "step": step,
                "learning_rate": scheduler.get_last_lr(),
            }
        )
