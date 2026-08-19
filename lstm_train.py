from pathlib import Path
import itertools
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from common import (
    RANDOM_STATE,
    compute_metrics,
    evaluate_by_minute_bucket as common_evaluate_by_minute_bucket,
    print_bucket_rows,
    print_metrics,
    save_calibration_plot,
)


DATA_FILE = Path("results/lstm_data.npz")
MODEL_FILE = Path("models/lstm_model.pt")
PREDICTIONS_FILE = Path("results/lstm_predictions.npz")
RESULTS_FILE = Path("results/lstm_grid_search_results.csv")
CALIBRATION_PLOT_FILE = Path("figures/lstm_calibration_curve.png")

BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 0.001


GRID = {
    "hidden_size": [64, 128],
    "num_layers": [1, 2],
    "dropout": [0.10, 0.25, 0.35],
    "weight_decay": [0.0, 1e-4],
}


torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class LSTMEveryMinute(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)
        return logits


def make_loader(X, y, mask, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def masked_loss(logits, y, mask, loss_fn):
    y_seq = y.unsqueeze(1).expand_as(logits)
    raw_loss = loss_fn(logits, y_seq)
    loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


def collect_predictions(model, loader, device):
    """Flatten every valid (match, minute) prediction in a loader to 1D arrays."""
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X, y, mask in loader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            y_seq = y.unsqueeze(1).expand_as(probs)

            valid_probs = probs[mask == 1].cpu().numpy()
            valid_targets = y_seq[mask == 1].cpu().numpy()

            all_probs.append(valid_probs)
            all_targets.append(valid_targets)

    return np.concatenate(all_probs), np.concatenate(all_targets)


def evaluate(model, loader, device):
    probs, targets = collect_predictions(model, loader, device)
    return compute_metrics(targets, probs)


def evaluate_final_minute(model, X, y, mask, device):
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        probs_all = torch.sigmoid(model(X_t)).cpu().numpy()

    probs = []
    targets = []

    for i in range(len(X)):
        valid_len = int(mask[i].sum())

        if valid_len == 0:
            continue

        final_idx = valid_len - 1
        probs.append(probs_all[i, final_idx])
        targets.append(y[i])

    return compute_metrics(np.array(targets), np.array(probs))


def evaluate_by_minute_bucket(model, X, y, mask, device):
    """Bucket LSTM per-minute predictions using the same buckets/metrics the
    XGBoost script uses (common.evaluate_by_minute_bucket), by first
    flattening the masked (match, minute) grid into a long dataframe.
    """
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        probs_all = torch.sigmoid(model(X_t)).cpu().numpy()

    n_matches, max_len = mask.shape
    minutes = np.tile(np.arange(1, max_len + 1), (n_matches, 1))
    targets = np.repeat(y[:, None], max_len, axis=1)

    valid = mask == 1

    pred_df = pd.DataFrame({
        "minute": minutes[valid],
        "target": targets[valid],
        "prob": probs_all[valid],
    })

    return common_evaluate_by_minute_bucket(pred_df, prob_col="prob")


def predict_all(model, X, device):
    model.eval()

    all_probs = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end = start + BATCH_SIZE
            X_t = torch.tensor(X[start:end], dtype=torch.float32).to(device)
            probs = torch.sigmoid(model(X_t)).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def get_test_ids(data):
    if "test_ids" in data:
        return data["test_ids"]

    if "match_ids_test" in data:
        return data["match_ids_test"]

    return np.array([f"test_match_{i}" for i in range(len(data["y_test"]))])


def config_iterator():
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def train_one_config(
    config,
    input_size,
    train_loader,
    val_loader,
    device,
):
    model = LSTMEveryMinute(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=config["weight_decay"],
    )

    best_val_log_loss = float("inf")
    best_val_auc = None
    best_val_accuracy = None
    best_epoch = 0
    bad_epochs = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for X, y, mask in train_loader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = masked_loss(logits, y, mask, loss_fn)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_log_loss={val_metrics['log_loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["log_loss"] < best_val_log_loss:
            best_val_log_loss = val_metrics["log_loss"]
            best_val_auc = val_metrics["auc"]
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            bad_epochs = 0

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_log_loss": best_val_log_loss,
        "best_val_accuracy": best_val_accuracy,
    }


def save_results_csv(results):
    if not results:
        return

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main():
    data = np.load(DATA_FILE, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    mask_train = data["mask_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    mask_val = data["mask_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    mask_test = data["mask_test"]

    test_ids = get_test_ids(data)

    input_size = X_train.shape[2]

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("Input features:", input_size)

    baseline = max(y_train.mean(), 1 - y_train.mean())
    print("Majority-class baseline accuracy:", round(float(baseline), 4))

    train_loader = make_loader(X_train, y_train, mask_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, mask_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, mask_test, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    configs = list(config_iterator())
    print("Total configs:", len(configs))

    results = []

    best_model = None
    best_config = None
    best_val_log_loss = float("inf")

    for i, config in enumerate(configs, start=1):
        print("\n" + "=" * 80)
        print(f"Config {i}/{len(configs)}")
        print(config)

        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        result = train_one_config(
            config=config,
            input_size=input_size,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        row = {
            **config,
            "best_epoch": result["best_epoch"],
            "val_auc": result["best_val_auc"],
            "val_log_loss": result["best_val_log_loss"],
            "val_accuracy": result["best_val_accuracy"],
        }

        results.append(row)
        save_results_csv(results)

        print("Saved current grid results to:", RESULTS_FILE)

        if result["best_val_log_loss"] < best_val_log_loss:
            best_val_log_loss = result["best_val_log_loss"]
            best_model = result["model"]
            best_config = config

    print("\n" + "=" * 80)
    print("Best config")
    print(best_config)
    print("Best validation log loss:", round(float(best_val_log_loss), 4))

    print("\nFinal evaluation")

    test_probs_all, test_targets_all = collect_predictions(best_model, test_loader, device)
    test_metrics_all = compute_metrics(test_targets_all, test_probs_all)
    print_metrics("Test metrics across all valid minutes", test_metrics_all)

    brier = save_calibration_plot(test_targets_all, test_probs_all, CALIBRATION_PLOT_FILE, "LSTM win probability")
    print("Saved calibration curve to:", CALIBRATION_PLOT_FILE, f"(Brier={brier:.4f})")

    test_metrics_final = evaluate_final_minute(
        best_model,
        X_test,
        y_test,
        mask_test,
        device,
    )
    print_metrics("Test metrics using final observed minute only", test_metrics_final)

    print()
    print("Test metrics by minute bucket")

    bucket_rows = evaluate_by_minute_bucket(
        best_model,
        X_test,
        y_test,
        mask_test,
        device,
    )
    print_bucket_rows(bucket_rows)

    test_probs = predict_all(best_model, X_test, device)

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        PREDICTIONS_FILE,
        probs_test=test_probs,
        y_test=y_test,
        mask_test=mask_test,
        match_ids_test=test_ids,
        best_hidden_size=best_config["hidden_size"],
        best_num_layers=best_config["num_layers"],
        best_dropout=best_config["dropout"],
        best_weight_decay=best_config["weight_decay"],
    )

    save_dict = {
        "model_state_dict": best_model.state_dict(),
        "input_size": input_size,
        "hidden_size": best_config["hidden_size"],
        "num_layers": best_config["num_layers"],
        "dropout": best_config["dropout"],
        "weight_decay": best_config["weight_decay"],
        "learning_rate": LEARNING_RATE,
        "test_auc_all_minutes": test_metrics_all["auc"],
        "test_log_loss_all_minutes": test_metrics_all["log_loss"],
        "test_accuracy_all_minutes": test_metrics_all["accuracy"],
        "test_brier_all_minutes": test_metrics_all["brier"],
    }

    if "feature_names" in data:
        save_dict["feature_names"] = data["feature_names"]

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, MODEL_FILE)

    print("\nSaved best model to:", MODEL_FILE)
    print("Saved predictions to:", PREDICTIONS_FILE)
    print("Saved grid search results to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
