import os
from pathlib import Path

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from training_neural_nets_pytorch_lighting.net_model_pytorch_lighting import (
    TextGenerativeNetModelPLighting,
    MetricTrackerCallback,
)

from pytorch_lightning.callbacks import (
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    EarlyStopping,
)
from pytorch_lightning.tuner import Tuner

import logging
import random


def seed_everything(seed=42):
    def force_cudnn_initialization():
        s = 32
        dev = torch.device("cuda")
        torch.nn.functional.conv2d(
            torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True)
    torch.backends.opt_einsum.enabled = True

    torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = False

    force_cudnn_initialization()

    return DEVICE


def set_logging():
    # configure logging at the root level of Lightning
    logger = logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    # logger.addHandler(logging.FileHandler("pytorch_ligting_core2.log"))

    # disable complete any log to console
    # logging.getLogger("lightning.pytorch").disabled

    # configure logging on module level, redirect to file
    logger = logging.getLogger("lightning.pytorch.core").setLevel(logging.ERROR)
    # logger.addHandler(logging.FileHandler("pytorch_ligting_core.log"))


set_logging()


def remove_folder(folder):
    clean_folder(folder)

    if os.path.exists(folder):
        folder = Path(folder)
        folder.rmdir()


def clean_folder(folder):
    folder = Path(folder)

    if os.path.exists(folder):
        for item in folder.iterdir():
            if item.is_dir():
                remove_folder(item)
            else:
                item.unlink()
    else:
        print(f"Folder: {folder.name} does not exist")
        return


class TrainingPytorchLighting:
    def __init__(self, neural_net_model, text_data_module, optimizer_parameters,tensorboard_experiment_name='experiment',clean_folder_log=True):
        self.neural_net_model = neural_net_model
        self.text_data_module = text_data_module

        self.train_outputs_logs = "pl_training_logs"
        self.tensorboard_folder = "tensorboard_folder"
        self.tensorboard_experiment_name=tensorboard_experiment_name
        self.model_checkpoints = "training_output_checkpoints"

        self.optimizer_parameters = optimizer_parameters
        self.clean_folder_log=clean_folder_log

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_trainer_net_model(
        self,
        max_epochs,
        lr_finder_setting=False,
        tensorboard_start=False,
        checked_validation_epochs=None,
    ):
        # print('self.DEVICE.type:',self.DEVICE, self.DEVICE.type)
        if self.clean_folder_log:
            clean_folder(self.train_outputs_logs)

        # settings to try :
        # accelerator=self.DEVICE.type,
        if lr_finder_setting:
            return (
                pl.Trainer(
                    default_root_dir=self.train_outputs_logs,
                    accelerator="auto",
                    min_epochs=max_epochs,
                    max_epochs=50,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                    logger=False,
                    check_val_every_n_epoch=checked_validation_epochs,
                    detect_anomaly=True,
                    val_check_interval=None,
                    limit_val_batches=0.2,
                    gradient_clip_val=50,
                ),
                None,
            )

        metric_tracker = MetricTrackerCallback()

        if not tensorboard_start:
            return (
                pl.Trainer(
                    default_root_dir=self.train_outputs_logs,
                    accelerator="auto",
                    min_epochs=3,
                    max_epochs=max_epochs,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=True,
                    logger=False,
                    val_check_interval=None,
                    limit_val_batches=0.2,
                    check_val_every_n_epoch=checked_validation_epochs,
                    detect_anomaly=True,
                    gradient_clip_val=50,
                    callbacks=[
                        metric_tracker,
                        LearningRateFinder(
                            min_lr=1e-06,
                            max_lr=9e-01,
                            num_training_steps=100,
                            early_stop_threshold=None,
                            update_attr=True,
                        ),
                        StochasticWeightAveraging(swa_lrs=1e-2),
                    ],
                ),
                metric_tracker,
            )
        if self.clean_folder_log:
            # clean folders from previouse trainings
            clean_folder("tensorboard_folder/".join(self.tensorboard_experiment_name))
            clean_folder("training_output_checkpoints/model_checkpoint")

        logger = TensorBoardLogger(
            save_dir=self.tensorboard_folder,
            name=self.tensorboard_experiment_name,
            log_graph=True,
        )
        logger._log_graph = True
        # logger._default_hp_metric = None
        
        return (
            # StochasticWeightAveraging(swa_lrs=1e-2),
            # EarlyStopping(monitor="train_auroc_score",patience=12, mode="max"), #train_auroc_score
            # monitor="train_loss",

            pl.Trainer(
                default_root_dir=self.train_outputs_logs,
                accelerator="auto",
                min_epochs=3,
                max_epochs=max_epochs,
                enable_checkpointing=True,
                enable_model_summary=True,
                callbacks=[
                    metric_tracker,
                    ModelCheckpoint(
                        dirpath=self.model_checkpoints,
                        save_weights_only=False,
                        mode="min",
                        monitor="train_loss",
                        save_top_k=3,
                        save_last=True,
                    ),
                    LearningRateMonitor(logging_interval="step"),
                    LearningRateFinder(
                        min_lr=1e-08,
                        max_lr=1e-01,
                        num_training_steps=7000,
                        early_stop_threshold=None,
                        update_attr=True,
                    ),
                ],
                enable_progress_bar=True,
                log_every_n_steps=1,
                logger=logger,
                check_val_every_n_epoch=checked_validation_epochs,
                # detect_anomaly=True,
                gradient_clip_val=10,
            ),
            metric_tracker,
        )

    # TO DO
    def any_lightning_module_function_or_hook(self):
        self.logger = TensorBoardLogger(
            save_dir=self.tensorboard_folder, name="training_best_model", log_graph=True
        )

        tensorboard_logger = self.logger.experiment

        prototype_array = torch.Tensor(32, 1, 28, 27)
        tensorboard_logger.log_graph(model=self, input_array=prototype_array)

    def train_net_model(self, max_epochs):
        # print('tensorboard_start: ',tensorboard_start)
        self.trainer_network_model, metric_tracker = self.get_trainer_net_model(
            max_epochs, tensorboard_start=True, checked_validation_epochs=None
        )

        self.text_classification_model = TextGenerativeNetModelPLighting(
            model_name=self.neural_net_model.to(self.DEVICE),
            optimizer_hparams=self.optimizer_parameters,
        )

        self.trainer_network_model.fit(
            model=self.text_classification_model, datamodule=self.text_data_module
        )

        # print('best_model_path: ', trainer.checkpoint_callback.best_model_path)
        # model = TextGenerativeNetModelPLighting.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # losses_trainning,accuracies_trainning,f1_scores_trainning,auroc_scores_trainning=metric_tracker.get_all_trainning_evaluations()
        (
            losses_trainning_score,
            accuracy_trainning_score,
            f1_score_trainning_score,
        ) = metric_tracker.get_trainning_evaluations_scores()

        # self.trainer_network_model.logger.log_metrics()

        results_str = "Train set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t  Average f1_score: {:.4f}% \n".format(
            np.around(losses_trainning_score, decimals=3),
            np.around(accuracy_trainning_score, decimals=3) * 100,
            np.around(f1_score_trainning_score, decimals=3) * 100,
        )

        self.trainer_network_model.logger.log_hyperparams(
            {"training_results:": results_str}
        )
        results_str = f"training_results: {results_str}"
        self.trainer_network_model.logger.log_metrics(metrics={results_str: 0})

        return results_str

    def test_net_model(self):
        test_result = self.trainer_network_model.test(
            model=self.text_classification_model,
            datamodule=self.text_data_module,
            verbose=True,
        )
        test_result = test_result.pop()

        results_str = "Test set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t  Average f1_score: {:.4f}% \n".format(
            np.around(test_result["test_loss_epoch"], decimals=3),
            np.around(test_result["test_acc_epoch"], decimals=3) * 100,
            np.around(test_result["test_f1_score_epoch"], decimals=3) * 100,
        )

        self.trainer_network_model.logger.log_hyperparams(
            {"testing_results:": results_str}
        )
        results_str = f"testing_results: {results_str}"
        self.trainer_network_model.logger.log_metrics(metrics={results_str: 0})

        return results_str

    def get_text_classification_model(self):
        return self.text_classification_model

    # it gives error : Failed to compute suggestion for learning rate because there are not enough points. Increase the loop iteration limits or the size of your dataset/dataloader.
    def find_learning_rate(self, max_epochs=50, tensorboard_start=True, plot_lr=False):
        model_vehicles = TextGenerativeNetModelPLighting(
            model_name=self.neural_net_model.to(self.DEVICE),
            optimizer_hparams=self.optimizer_parameters,
        )

        trainer_network_model, _ = self.get_trainer_net_model(
            max_epochs, lr_finder_setting=True
        )

        tuner = Tuner(trainer_network_model)
        if tensorboard_start:
            # Run learning rate finder
            lr_finder = tuner.lr_find(
                model=model_vehicles,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.test_loader,
                min_lr=1e-07,
                max_lr=9e-01,
                num_training=10000,
                update_attr=True,
            )
        else:
            # Run learning rate finder
            lr_finder = tuner.lr_find(
                model=model_vehicles,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.test_loader,
                min_lr=1e-07,
                max_lr=9e-01,
                num_training=3500,
                update_attr=True,
            )

        if plot_lr:
            lr_finder.plot()

        print("lr_finder.suggestion(): ", lr_finder.suggestion())
        return lr_finder.suggestion()
