import os
import time
import numpy as np
import torch
import torch.nn as nn

from src.utils.metrics import masked_mae, masked_rmse, masked_mape


class HybridEngine:

    def __init__(
        self, device, model, dataloader, loss_fn,
        optimizer, scheduler, clip_grad_value,
        max_epochs, patience, log_dir, logger
    ):
        self.device = device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.patience = patience
        self.clip_grad_value = clip_grad_value
        self.logger = logger
        self.log_dir = log_dir

        self.best_val = np.inf
        self.best_state = None
        self.wait = 0

        # Disable AMP for stability (ChebConv + DynamicAdj is unstable in AMP)
        self.autocast_enabled = False


    # --------------------------------------------------------
    # TRAIN ONE EPOCH
    # --------------------------------------------------------
    def train_one_epoch(self):
        self.model.train()
        loader = self.dataloader["train_loader"]

        loss_arr, mae_arr, rmse_arr, mape_arr = [], [], [], []
        t0 = time.time()

        for x, y in loader.get_iterator():

            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            y = torch.tensor(y, device=self.device, dtype=torch.float32)

            # TOD: [B,T]
            t_index = x[:, :, 0, 1]

            # reorder -> [B, C, T, N]
            x = x.permute(0, 3, 1, 2)

            self.optimizer.zero_grad()

            # ------------------------------------------------
            # Forward pass (AMP disabled for safety)
            # ------------------------------------------------
            with torch.cuda.amp.autocast(enabled=False):
                y_pred = self.model(x, t_index)

                loss = self.loss_fn(y_pred, y)
                mae = masked_mae(y_pred, y)
                rmse = masked_rmse(y_pred, y)
                mape = masked_mape(y_pred, y)

            # ------------------------------------------------
            # Backward pass (safe mode)
            # ------------------------------------------------
            loss.backward()

            # ❗ CRITICAL FIX — Clip gradients to avoid NaNs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_value)

            self.optimizer.step()

            loss_arr.append(loss.item())
            mae_arr.append(mae.item())
            rmse_arr.append(rmse.item())
            mape_arr.append(mape.item())

        return (
            np.mean(loss_arr), np.mean(mae_arr),
            np.mean(rmse_arr), np.mean(mape_arr),
            time.time() - t0
        )


    # --------------------------------------------------------
    # EVAL ONE EPOCH
    # --------------------------------------------------------
    def eval_one_epoch(self):
        self.model.eval()
        loader = self.dataloader["val_loader"]

        loss_arr, mae_arr, rmse_arr, mape_arr = [], [], [], []
        t0 = time.time()

        with torch.no_grad():
            for x, y in loader.get_iterator():
                x = torch.tensor(x, device=self.device, dtype=torch.float32)
                y = torch.tensor(y, device=self.device, dtype=torch.float32)

                t_index = x[:, :, 0, 1]
                x = x.permute(0, 3, 1, 2)

                with torch.cuda.amp.autocast(enabled=False):
                    y_pred = self.model(x, t_index)

                    loss  = self.loss_fn(y_pred, y)
                    mae   = masked_mae(y_pred, y)
                    rmse  = masked_rmse(y_pred, y)
                    mape  = masked_mape(y_pred, y)

                loss_arr.append(loss.item())
                mae_arr.append(mae.item())
                rmse_arr.append(rmse.item())
                mape_arr.append(mape.item())

        return (
            np.mean(loss_arr), np.mean(mae_arr),
            np.mean(rmse_arr), np.mean(mape_arr),
            time.time() - t0
        )


    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    def train(self):
        self.logger.info("Starting HybridModel training...")

        ckpt_path = os.path.join(self.log_dir, "best_model.pt")

        for epoch in range(1, self.max_epochs + 1):

            tr_loss, tr_mae, tr_rmse, tr_mape, tr_time = self.train_one_epoch()
            val_loss, val_mae, val_rmse, val_mape, val_time = self.eval_one_epoch()

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"Train: MAE {tr_mae:.4f}, RMSE {tr_rmse:.4f}, MAPE {tr_mape:.2f}% | "
                f"Val: MAE {val_mae:.4f}, RMSE {val_rmse:.4f}, MAPE {val_mape:.2f}% | "
                f"Time: {tr_time:.1f}s + {val_time:.1f}s"
            )

            # -----------------------------
            # EARLY STOPPING
            # -----------------------------
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.wait = 0

                # save checkpoint
                self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                torch.save(self.best_state, ckpt_path)

                self.logger.info(f"  -> Val improved! Saving checkpoint (MAE: {val_mae:.4f})")

            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.logger.info("Early stopping triggered")
                    break

            self.scheduler.step()

        # Load best weights
        self.model.load_state_dict(self.best_state)
        self.logger.info("Training complete — loaded best model.")
