from pathlib import Path
import itertools
import csv

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


DATA_FILE = Path("lstm_data.npz")
MODEL_FILE = Path("lstm_model.pt")
PREDICTIONS_FILE = Path("lstm_predictions.npz")
RESULTS_FILE = Path("lstm_grid_search_results.csv")

BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 0.001
RANDOM_STATE = 101705


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


def evaluate(model, loader, device):
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

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(targets, probs),
        "log_loss": log_loss(targets, probs, labels=[0, 1]),
        "accuracy": accuracy_score(targets, preds),
    }


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

    probs = np.array(probs)
    targets = np.array(targets)
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(targets, probs),
        "log_loss": log_loss(targets, probs, labels=[0, 1]),
        "accuracy": accuracy_score(targets, preds),
    }


def evaluate_by_minute_bucket(model, X, y, mask, device):
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        probs_all = torch.sigmoid(model(X_t)).cpu().numpy()

    rows = []

    buckets = [
        (1, 5),
        (6, 10),
        (11, 15),
        (16, 20),
        (21, 25),
        (26, 30),
        (31, 45),
    ]

    for start, end in buckets:
        probs = []
        targets = []

        for i in range(len(X)):
            for t in range(start - 1, min(end, X.shape[1])):
                if mask[i, t] == 1:
                    probs.append(probs_all[i, t])
                    targets.append(y[i])

        if len(probs) == 0:
            continue

        probs = np.array(probs)
        targets = np.array(targets)
        preds = (probs >= 0.5).astype(int)

        if len(np.unique(targets)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(targets, probs)

        rows.append({
            "minutes": f"{start}-{end}",
            "rows": len(targets),
            "auc": auc,
            "log_loss": log_loss(targets, probs, labels=[0, 1]),
            "accuracy": accuracy_score(targets, preds),
        })

    return rows


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

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def print_metrics(name, metrics):
    print(name)
    print("AUC:", round(float(metrics["auc"]), 4))
    print("Log loss:", round(float(metrics["log_loss"]), 4))
    print("Accuracy:", round(float(metrics["accuracy"]), 4))


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

    test_metrics_all = evaluate(best_model, test_loader, device)
    print_metrics("\nTest metrics across all valid minutes", test_metrics_all)

    test_metrics_final = evaluate_final_minute(
        best_model,
        X_test,
        y_test,
        mask_test,
        device,
    )
    print_metrics("\nTest metrics using final observed minute only", test_metrics_final)

    print("\nTest metrics by minute bucket")

    bucket_rows = evaluate_by_minute_bucket(
        best_model,
        X_test,
        y_test,
        mask_test,
        device,
    )

    for row in bucket_rows:
        auc_value = row["auc"]

        if np.isnan(auc_value):
            auc_text = "nan"
        else:
            auc_text = round(float(auc_value), 4)

        print(
            row["minutes"],
            "| rows:", row["rows"],
            "| auc:", auc_text,
            "| log_loss:", round(float(row["log_loss"]), 4),
            "| accuracy:", round(float(row["accuracy"]), 4),
        )

    test_probs = predict_all(best_model, X_test, device)

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
    }

    if "feature_names" in data:
        save_dict["feature_names"] = data["feature_names"]

    torch.save(save_dict, MODEL_FILE)

    print("\nSaved best model to:", MODEL_FILE)
    print("Saved predictions to:", PREDICTIONS_FILE)
    print("Saved grid search results to:", RESULTS_FILE)


if __name__ == "__main__":
    main()