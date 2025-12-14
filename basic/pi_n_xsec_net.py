import os
import time
import wandb
import torch
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import lightning as L

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

@dataclass
class TrainConfig:
    seed: int = 1438

    test_size: float = 0.3
    val_size_in_residual: float = 0.66
    split_random_state: int = 42

    clip_quantile: float = 0.96
    ebeam_fix_from: int = 8314
    ebeam_fix_to: int = 65671
    ebeam_fix_value: float = 5.754

    net_architecture: Tuple[int, ...] = (5,60,80,100,120,140,240,340,440,640,2000,1040,640,340,240,140,100,80,60,20,1)
    n_params = sum(x * y for x, y in zip(net_architecture[:-1], net_architecture[1:]))
    activation: str = "ReLU"

    batch_size: int = 256
    max_epochs: int = 100
    lr: float = 1e-3
    optim: str = "Adam"

    data_path: str = "../data/clasdb_pi_plus_n.txt"
    wandb_enabled: bool = True
    wandb_project: str = "pi_plus_n"
    wandb_save_dir: str = "./wandb_local_logs"
    run_name: str = f"pi_plus_n_layers_{len(net_architecture)}_params_{n_params//1000}k_v{int(time.time())}"
    run_dir: str = os.path.join(wandb_save_dir, wandb_project, run_name)

    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_cooldown: int = 3

    early_stopping: bool = True
    es_monitor: str = "val_loss"
    es_mode: str = "min"
    es_patience: int = 15
    es_min_delta: float = 0.0
    es_verbose: bool = False

    # test metrics policy
    test_each_epoch: bool = True
    test_every_n_epochs: int = 1
    run_final_test: bool = True

    accelerator: str = "gpu"
    devices: str = "auto"


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EpochOnlyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.disable = True
        return bar


class PiPlusNLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, cfg: TrainConfig, label_scaler: StandardScaler):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = nn.MSELoss()

        self.register_buffer("y_mean", torch.tensor(float(label_scaler.mean_[0]), dtype=torch.float32))
        self.register_buffer("y_scale", torch.tensor(float(label_scaler.scale_[0]), dtype=torch.float32))

        self._val_y: List[torch.Tensor] = []
        self._val_pred: List[torch.Tensor] = []

        self._test_dl: Optional[DataLoader] = None
        self.last_test: Dict[str, float] = {}

        try:
            self._optim_cls = getattr(torch.optim, cfg.optim)
        except AttributeError as e:
            raise ValueError(f"Unknown optimizer '{cfg.optim}'. Example: Adam, AdamW, SGD") from e

    def set_test_dataloader(self, dl: DataLoader) -> None:
        self._test_dl = dl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        opt = self._optim_cls(self.parameters(), lr=self.cfg.lr)
        sch = ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            factor=self.cfg.lr_factor,
            patience=self.cfg.lr_patience,
            cooldown=self.cfg.lr_cooldown,
            threshold=0.01,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def _rmse_scaled(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.loss_fn(pred, y))

    def _inv(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.y_scale + self.y_mean

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).reshape(-1)
        loss = self._rmse_scaled(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).reshape(-1)
        loss = self._rmse_scaled(pred, y)

        self._val_y.append(y.detach())
        self._val_pred.append(pred.detach())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _compute_mae_mse_orig_units(self, y_s: torch.Tensor, p_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self._inv(y_s)
        p = self._inv(p_s)
        mae = torch.mean(torch.abs(p - y))
        mse = torch.mean((p - y) ** 2)
        return mae, mse

    def _maybe_compute_test_each_epoch(self):
        if not self.cfg.test_each_epoch:
            return
        if self._test_dl is None:
            return
        if self.cfg.test_every_n_epochs <= 0:
            return
        if (self.current_epoch + 1) % self.cfg.test_every_n_epochs != 0:
            return

        self.model.eval()
        preds = []
        ys = []
        device = self.device

        with torch.no_grad():
            for xb, yb in self._test_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                out = self(xb).reshape(-1)
                preds.append(out)
                ys.append(yb)

        y_s = torch.cat(ys)
        p_s = torch.cat(preds)
        test_mae, test_mse = self._compute_mae_mse_orig_units(y_s, p_s)

        self.log("test_mae", test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mse", test_mse, prog_bar=True, on_step=False, on_epoch=True)

        self.last_test = {"mae": float(test_mae.detach().cpu().item()), "mse": float(test_mse.detach().cpu().item())}

    def on_validation_epoch_end(self):
        if self._val_y:
            y_s = torch.cat(self._val_y)
            p_s = torch.cat(self._val_pred)

            val_mae, val_mse = self._compute_mae_mse_orig_units(y_s, p_s)

            self.log("val_mae", val_mae, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_mse", val_mse, prog_bar=True, on_step=False, on_epoch=True)

            self._val_y.clear()
            self._val_pred.clear()

        self._maybe_compute_test_each_epoch()

    # ---- Support trainer.test() (final test) ----
    def on_test_epoch_start(self):
        self._test_y: List[torch.Tensor] = []
        self._test_pred: List[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).reshape(-1)
        self._test_y.append(y.detach())
        self._test_pred.append(pred.detach())
        return None

    def on_test_epoch_end(self):
        if not self._test_y:
            return
        y_s = torch.cat(self._test_y)
        p_s = torch.cat(self._test_pred)

        test_mae, test_mse = self._compute_mae_mse_orig_units(y_s, p_s)

        self.log("test_mae", test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mse", test_mse, prog_bar=True, on_step=False, on_epoch=True)

        self.last_test = {"mae": float(test_mae.detach().cpu().item()), "mse": float(test_mse.detach().cpu().item())}


class PiPlusNElectroproductionRegressor:
    FEATURE_COLUMNS: List[str] = ["Ebeam", "W", "Q2", "cos_theta", "cos_phi"]
    LABEL_COLUMN: str = "dsigma_dOmega"

    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()
        set_random_seed(self.cfg.seed)

        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()

        self._trainer: Optional[L.Trainer] = None
        self._pl_module: Optional[PiPlusNLightningModule] = None

        self.csv_logger = CSVLogger(save_dir=self.cfg.run_dir, name="csv")

        self.wandb_logger = None
        if self.cfg.wandb_enabled:
            wandb.login()
            self.wandb_logger = WandbLogger(
                project=self.cfg.wandb_project,
                save_dir=self.cfg.run_dir,
                name=self.cfg.run_name,
                log_model=False
            )

        self.ckpt_cb = ModelCheckpoint(
            dirpath=f"{self.cfg.run_dir}/checkpoints",
            filename="best-{epoch:03d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

    def load_and_prepare_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.data_path, delimiter="\t", header=None)
        df.columns = ["Ebeam", "W", "Q2", "cos_theta", "phi", "dsigma_dOmega", "error", "id"]

        df.loc[self.cfg.ebeam_fix_from:self.cfg.ebeam_fix_to, "Ebeam"] = self.cfg.ebeam_fix_value

        phi_rad = np.deg2rad(df["phi"].to_numpy(dtype=np.float64))
        df["phi"] = phi_rad
        df["cos_phi"] = np.cos(phi_rad)

        df = df.iloc[df[["Ebeam", "W", "Q2", "cos_theta", "phi"]].drop_duplicates().index]
        df = df.drop(columns=["id"])

        q = self.cfg.clip_quantile
        df = df[df["dsigma_dOmega"] <= df["dsigma_dOmega"].quantile(q)]
        df = df[df["error"] <= df["error"].quantile(q)]

        return df.reset_index(drop=True)

    def make_splits_and_dataloaders(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        X = df[self.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = df[self.LABEL_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

        X_train, X_res, y_train, y_res = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.split_random_state
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_res, y_res, test_size=self.cfg.val_size_in_residual, random_state=self.cfg.split_random_state
        )

        X_train = self.feature_scaler.fit_transform(X_train)
        X_val = self.feature_scaler.transform(X_val)
        X_test = self.feature_scaler.transform(X_test)

        y_train = self.label_scaler.fit_transform(y_train)
        y_val = self.label_scaler.transform(y_val)
        y_test = self.label_scaler.transform(y_test)

        def dl(Xa, ya, shuffle: bool) -> DataLoader:
            X_t = torch.tensor(Xa, dtype=torch.float32)
            y_t = torch.tensor(ya.reshape(-1), dtype=torch.float32)
            return DataLoader(TensorDataset(X_t, y_t), batch_size=self.cfg.batch_size, shuffle=shuffle)

        return dl(X_train, y_train, True), dl(X_val, y_val, False), dl(X_test, y_test, False)

    def _build_network(self) -> nn.Module:
        arch = list(self.cfg.net_architecture)
        if arch[0] != len(self.FEATURE_COLUMNS):
            raise ValueError(f"net_architecture[0]={arch[0]} but expected {len(self.FEATURE_COLUMNS)}")

        try:
            act_cls = getattr(nn, self.cfg.activation)
        except AttributeError as e:
            raise ValueError(f"Unknown activation '{self.cfg.activation}'") from e

        layers: List[nn.Module] = []
        for i in range(1, len(arch)):
            layers.append(nn.Linear(arch[i - 1], arch[i]))
            if i != len(arch) - 1:
                layers.append(act_cls())
        return nn.Sequential(*layers)

    def fit(self) -> Dict[str, float]:
        df = self.load_and_prepare_dataframe()
        train_dl, val_dl, test_dl = self.make_splits_and_dataloaders(df)

        net = self._build_network()
        pl_module = PiPlusNLightningModule(net, self.cfg, label_scaler=self.label_scaler)
        pl_module.set_test_dataloader(test_dl)

        callbacks = [EpochOnlyProgressBar(refresh_rate=1)]
        if self.cfg.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.cfg.es_monitor,
                    mode=self.cfg.es_mode,
                    patience=self.cfg.es_patience,
                    min_delta=self.cfg.es_min_delta,
                    verbose=self.cfg.es_verbose,
                )
            )

        trainer = L.Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator=self.cfg.accelerator,
            devices=self.cfg.devices,
            log_every_n_steps=10,
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=[self.csv_logger, self.wandb_logger] if self.wandb_logger else self.csv_logger
        )

        trainer.fit(model=pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        if self.cfg.run_final_test:
            trainer.test(model=pl_module, dataloaders=test_dl, verbose=False)
            final = pl_module.last_test
        else:
            # If we didn't run trainer.test(), but test_each_epoch ran, we still have last_test from epoch-end.
            final = pl_module.last_test

        # fallback if for some reason no test was computed anywhere
        if not final:
            y_scaled, pred_scaled = self._collect_test_arrays(pl_module, test_dl)
            y = self.label_scaler.inverse_transform(y_scaled)
            pred = self.label_scaler.inverse_transform(pred_scaled)
            final = {"mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred))}

        self._trainer = trainer
        self._pl_module = pl_module

        if self.cfg.wandb_enabled:
            wandb.finish()

        return {"mae": float(final["mae"]), "mse": float(final["mse"])}

    @staticmethod
    def _collect_test_arrays(pl_module: PiPlusNLightningModule, test_dl: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        pl_module.model.eval()
        ys = []
        ps = []
        device = pl_module.device

        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                out = pl_module(xb).reshape(-1)
                ys.append(yb.detach().cpu().numpy().reshape(-1, 1))
                ps.append(out.detach().cpu().numpy().reshape(-1, 1))

        return np.vstack(ys), np.vstack(ps)

    @torch.no_grad()
    def predict_df(self, df_features: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        if self._pl_module is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")

        X = df_features[self.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        Xs = self.feature_scaler.transform(X)

        X_t = torch.tensor(Xs, dtype=torch.float32)
        dl = DataLoader(TensorDataset(X_t, torch.zeros(len(X_t))), batch_size=batch_size, shuffle=False)

        device = self._pl_module.device
        preds = []

        self._pl_module.model.eval()
        for xb, _ in dl:
            xb = xb.to(device)
            out = self._pl_module(xb).detach().cpu().numpy().reshape(-1, 1)
            preds.append(out)

        preds_s = np.vstack(preds)
        preds_orig = self.label_scaler.inverse_transform(preds_s).reshape(-1)
        return preds_orig


def _main():
    cfg = TrainConfig()
    model = PiPlusNElectroproductionRegressor(cfg)
    metrics = model.fit()
    print(f"Done. Test MAE = {metrics['mae']:.6f}, Test MSE = {metrics['mse']:.6f}")


if __name__ == "__main__":
    _main()
