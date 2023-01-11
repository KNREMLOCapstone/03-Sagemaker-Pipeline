import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder


sm_output_dir = Path(os.environ.get("SM_OUTPUT_DIR"))
sm_model_dir = Path(os.environ.get("SM_MODEL_DIR"))
num_cpus = int(os.environ.get("SM_NUM_CPUS"))

ml_root = Path("/opt/ml")

git_path = ml_root / "sagemaker-intelimage"

dvc_repo_url = os.environ.get("DVC_REPO_URL")
dvc_branch = os.environ.get("DVC_BRANCH")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy = Accuracy(task="multiclass", num_classes=6)


def get_training_env():
    sm_training_env = os.environ.get("SM_TRAINING_ENV")
    sm_training_env = json.loads(sm_training_env)

    return sm_training_env


class LitResnet(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f"train/loss", loss, prog_bar=True)
        self.log(f"train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}


class IntelImgClfDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)

        # data transformations
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.data_train.classes)

    @property
    def classes(self):
        return self.data_train.classes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            trainset = ImageFolder(self.data_dir / "train", transform=self.transforms)
            testset = ImageFolder(self.data_dir / "test", transform=self.transforms)
            valset = ImageFolder(self.data_dir / "val", transform=self.transforms)

            self.data_train, self.data_test, self.data_val = trainset, testset, valset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def train_and_evaluate(model, datamodule, sm_training_env, output_dir):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"]
    )
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=[tb_logger],
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    idx_to_class = {k: v for v, k in datamodule.data_train.class_to_idx.items()}
    model.idx_to_class = idx_to_class

    # per class accuracy
    confusion_matrix = torch.zeros(datamodule.num_classes, datamodule.num_classes)
    with torch.no_grad():
        for i, (images, targets) in enumerate(datamodule.test_dataloader()):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    acc_per_class = {
        idx_to_class[idx]: val.item() * 100
        for idx, val in enumerate(confusion_matrix.diag() / confusion_matrix.sum(1))
    }
    print(acc_per_class)

    with open(output_dir / "accuracy_per_class.json", "w") as outfile:
        json.dump(acc_per_class, outfile)


def save_scripted_model(model, output_dir):
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, output_dir / "model.scripted.pt")


def clone_dvc_git_repo():
    print(f":: Configure git to pull authenticated from CodeCommit")
    print(f":: Cloning repo: {dvc_repo_url}, git branch: {dvc_branch}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--branch", dvc_branch, dvc_repo_url, git_path]
    )


def dvc_pull():
    print(":: Running dvc pull command")
    os.chdir(git_path)

    print(f":: Pull from DVC")
    subprocess.check_call(["dvc", "pull"])


if __name__ == "__main__":
    clone_dvc_git_repo()
    dvc_pull()

    img_dset = ImageFolder(git_path / "dataset" / "train")

    print(":: Classnames: ", img_dset.classes)

    datamodule = IntelImgClfDataModule(
        data_dir=(git_path / "dataset").absolute(), num_workers=num_cpus
    )
    datamodule.setup()

    model = LitResnet(num_classes=datamodule.num_classes)
    model = model.to(device)

    sm_training_env = get_training_env()

    print(":: Training ...")
    train_and_evaluate(model, datamodule, sm_training_env, sm_model_dir)

    print(":: Saving Scripted Model")
    save_scripted_model(model, sm_model_dir)