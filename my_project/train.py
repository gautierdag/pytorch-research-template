import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from my_project.config import MyProjectConfig
from my_project.dataset import MyProjectDataModule
from my_project.model import MyProjectModel


def train(cfg: MyProjectConfig) -> None:
    # ensure wandb cache is in output dir (this gets cluttered rapidly otherwise)
    os.environ["WANDB_CACHE_DIR"] = str(cfg.paths.output_dir / "wandb")

    run_name = f"{cfg.run_name}_{cfg.model.bert_model}_{cfg.seed}"
    checkpoint_path = cfg.paths.output_dir / "checkpoints" / run_name
    unique_run_id = f"{wandb.util.generate_id()}_{cfg.seed}"

    wandb_logger = WandbLogger(
        name=run_name,
        project="my_project",
        entity="my_wandb_entity",
        config=cfg,
        save_dir=cfg.paths.output_dir,
        mode="disabled" if cfg.fast else "online",
        id=unique_run_id,
        group=cfg.run_name,
    )
    my_project_datamodule = MyProjectDataModule(
        data_dir_path=cfg.paths.input_dir,
        output_dir_path=cfg.paths.output_dir,
        batch_size=cfg.train.batch_size,
        bert_model_name=cfg.model.bert_model,
        max_seq_length=cfg.model.max_seq_length,
        num_workers=cfg.num_workers,
    )

    model = MyProjectModel(
        learning_rate=cfg.model.learning_rate,
        bert_model_name=cfg.model.bert_model,
        max_seq_length=cfg.model.max_seq_length,
        cache_dir=cfg.paths.output_dir / "bert-models",
    )

    print("Creating Trainer")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        every_n_epochs=1,
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger,
        accelerator="gpu",
        devices=cfg.gpus,
        callbacks=[checkpoint_callback],
        strategy="ddp",
    )

    trainer.fit(model, datamodule=my_project_datamodule)
    trainer.test(
        model,
        datamodule=my_project_datamodule,
        ckpt_path=checkpoint_callback.best_model_path,
    )

    # Save the model as a WANDB artifact
    if cfg.upload_model:
        art = wandb.Artifact("my_project_model", type="model")
        art.add_file(checkpoint_callback.best_model_path)
        wandb.log_artifact(art)

    wandb.finish()
