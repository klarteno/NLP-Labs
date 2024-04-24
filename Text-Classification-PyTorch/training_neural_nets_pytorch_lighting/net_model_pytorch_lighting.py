# from torchmetrics import classification
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import functional
import torchmetrics

import torch.optim as optim
import torch.nn as nn
import torch
import pytorch_lightning as pl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pl.seed_everything(42)


class TextGenerativeNetModelPLighting(pl.LightningModule):
    def __init__(self, model_name, optimizer_hparams):
        """
        Inputs:
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=['model_name'])

        # Create model
        self.model = model_name

        self.loss_module = nn.CrossEntropyLoss()
        self.learning_rate = self.hparams.optimizer_hparams["optimizer"][
            "learning_rate"
        ]

        self.start_logging = self.hparams.optimizer_hparams["start_logging"]
        self.batch_size = self.hparams.optimizer_hparams["batch_size"]

        self.example_input_array = (
            (torch.zeros((312), dtype=torch.long)),
            (torch.zeros((8), dtype=torch.long)),
        ) # torch.zeros((2,52,175), dtype=torch.float32).to('cpu')
        # self.example_input_array = {'text':torch.zeros((3828), dtype=torch.long),'offsets': torch.zeros((2),dtype=torch.long) }

    def forward(self, text, offsets):
        # Forward function that is run when visualizing the graph
        return self.model(text, offsets)

    def configure_optimizers(self):
        optimizer_params = self.hparams.optimizer_hparams["optimizer"]

        lr = optimizer_params["learning_rate"]
        params_net_model = filter(lambda p: p.requires_grad, self.parameters())

        if optimizer_params["name"] == "AdamW":
            optimizer = optim.AdamW(params=params_net_model, lr=lr)

        elif optimizer_params["name"] == "RAdam":
            optimizer = optim.RAdam(params=params_net_model, lr=lr)

        elif optimizer_params["name"] == "NAdam":
            optimizer = optim.NAdam(params=params_net_model, lr=lr)

        else:
            assert False, f'Unknown optimizer: "{optimizer_params["name"]}"'

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 =10)

        # return [optimizer], [scheduler]

        # if type(self.hparams.optimizer_hparams['lr_scheduler']) is torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:

        lr_scheduler_params = self.hparams.optimizer_hparams["lr_scheduler"]

        if (
            self.hparams.optimizer_hparams["lr_scheduler"]["name"]
            == "CosineAnnealingWarmRestarts"
        ):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=lr_scheduler_params["T_0"],
                        eta_min=lr_scheduler_params["eta_min"],
                    ),
                    "interval": "epoch",
                    "frequency": 2,
                },
            }

        elif self.hparams.optimizer_hparams["lr_scheduler"]["name"] == "OneCycleLR":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=lr_scheduler_params["max_lr"],
                        steps_per_epoch=lr_scheduler_params["steps_per_epoch"],
                        epochs=lr_scheduler_params["epochs"],
                    ),
                    "interval": "step",
                    "frequency": 30,
                },
            }

        else:
            print("scheduler unknown")

    def training_step(self, batch, batch_idx):
        text, label, offsets = batch
        # batch_x, batch_y = batch_x.contiguous().to(DEVICE, non_blocking=True), batch_y.contiguous().to(DEVICE, non_blocking=True)
        # batch_x, batch_y = batch_x.contiguous(), batch_y.contiguous()
        # print('batch_x1: ',batch_x.device)
        predicted_label = self(text, offsets)
        predicted_label = predicted_label.view(-1)

        loss = self.loss_module(predicted_label, label.float())
        num_classes = 4

        acc = functional.accuracy(
            predicted_label, label, task="multiclass", num_classes=num_classes
        )
        f1_score = functional.f1_score(
            predicted_label, label, task="multiclass", num_classes=num_classes
        )

        if self.start_logging:
            # Logs the accuracy per epoch to tensorboard ()
            self.log(
                "train_acc",
                acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
            self.log(
                "train_f1_score",
                f1_score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )

            # return {'acc':acc,'loss':loss,'f1_score':f1_score,'auroc_score':auroc_score}

        return loss

    def test_step(self, batch, batch_idx):
        text, label, offsets = batch
        # batch_x, batch_y = batch_x.contiguous().to(DEVICE, non_blocking=True), batch_y.contiguous().to(DEVICE, non_blocking=True)
        # batch_x, batch_y = batch_x.contiguous(), batch_y.contiguous()

        predicted_label = self.model(text, offsets)
        predicted_label = predicted_label.view(-1)

        loss = self.loss_module(predicted_label, label.float())
        num_classes = 4

        acc = functional.accuracy(
            predicted_label, label, task="multiclass", num_classes=num_classes
        )
        f1_score = functional.f1_score(
            predicted_label, label, task="multiclass", num_classes=num_classes
        )

        if self.start_logging:
            self.log(
                "test_acc",
                acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
            self.log(
                "test_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
            self.log(
                "test_f1_score",
                f1_score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )

            return acc
            # return {'acc':acc,'f1_score':f1_score,'auroc_score':auroc_score}

        return acc


from pytorch_lightning.callbacks import Callback


class MetricTrackerCallback(Callback):
    def __init__(self):
        self.losses_trainning = []
        self.accuracies_trainning = []
        self.f1_scores_trainning = []

        self.accuracies_validation = []
        self.f1_scores_validation = []

        self.device_for_metrics = "cpu"
        kwargs = {"compute_on_cpu": True}  # if use_cuda else {}

        self.mean_metric = torchmetrics.MeanMetric(nan_strategy="warn", **kwargs).to(
            self.device_for_metrics
        )

    def get_all_trainning_evaluations(self):
        return (
            self.losses_trainning,
            self.accuracies_trainning,
            self.f1_scores_trainning,
        )

    def get_trainning_evaluations_scores(self):
        losses_trainning = self.compute_mean(self.losses_trainning)
        accuracy_trainning = self.compute_mean(self.accuracies_trainning)
        f1_score_trainning = self.compute_mean(self.f1_scores_trainning)

        return (
            losses_trainning.item(),
            accuracy_trainning.item(),
            f1_score_trainning.item(),
        )

    def get_all_validation_evaluations(self):
        return (self.accuracies_validation, self.f1_scores_validation)

    def get_validation_evaluations_scores(self):
        accuracy_validation = self.compute_mean(self.accuracies_validation)
        f1_score_validation = self.compute_mean(self.f1_scores_validation)

        return (accuracy_validation.item(), f1_score_validation.item())

    def compute_mean(self, values):
        self.mean_metric.update(values)
        mean_value = self.mean_metric.compute()
        self.mean_metric.reset()

        return mean_value

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.losses_trainning.append(trainer.logged_metrics["train_loss_epoch"])
        self.accuracies_trainning.append(trainer.logged_metrics["train_acc_epoch"])
        self.f1_scores_trainning.append(trainer.logged_metrics["train_f1_score_epoch"])

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print("trainer.current_epoch: ", trainer.current_epoch)
