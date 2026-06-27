import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def worker_init_fn_affinity(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))


class UniversalDataModule(LightningDataModule):
    def __init__(
        self,
        train,
        val,
        test,
        batch_size_train,
        batch_size_val,
        batch_size_test,
        collate_fn: None,
        dset_version="one",  # two,three
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor=None,
        worker_init_fn=None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["train", "val", "test", "collate_fn"])

        self.data_train = train
        self.data_val = val
        self.data_test = test
        self.collate_fn = collate_fn

        self.dset_version = dset_version

        self.data_train.dset_version = dset_version
        self.data_val.dset_version = dset_version
        self.data_test.dset_version = dset_version

        self.worker_init_fn = None

        if num_workers > 0:
            self.worker_init_fn = worker_init_fn_affinity

        if num_workers == 0:
            self.hparams.prefetch_factor = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size_train % self.trainer.world_size != 0:
                raise RuntimeError(f"Batch size ({self.hparams.batch_size_train}) is not divisible by the number of devices ({self.trainer.world_size}).")
            self.batch_size_per_device = self.hparams.batch_size_train // self.trainer.world_size
            print(f" batch size per device: {self.batch_size_per_device}")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.batch_size_train,
            prefetch_factor=self.hparams.prefetch_factor,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.batch_size_val,
            prefetch_factor=self.hparams.prefetch_factor,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            batch_size=self.hparams.batch_size_test,
            worker_init_fn=self.worker_init_fn,
        )
