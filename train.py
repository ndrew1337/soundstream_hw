import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import SoundStreamTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    discriminator = instantiate(config.discriminator).to(device)
    logger.info(model)
    logger.info(discriminator)

    # get function handles of loss and metrics
    g_criterion = instantiate(config.g_loss).to(device)
    d_criterion = instantiate(config.d_loss).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    g_params = filter(lambda p: p.requires_grad, model.parameters())
    d_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    g_optimizer = instantiate(config.g_optimizer, params=g_params)
    d_optimizer = instantiate(config.d_optimizer, params=d_params)
    g_lr_scheduler = instantiate(config.g_lr_scheduler, optimizer=g_optimizer)
    d_lr_scheduler = instantiate(config.d_lr_scheduler, optimizer=d_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = SoundStreamTrainer(
        model=model,
        discriminator=discriminator,
        g_criterion=g_criterion,
        d_criterion=d_criterion,
        metrics=metrics,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_lr_scheduler=g_lr_scheduler,
        d_lr_scheduler=d_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
