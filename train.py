import os, math, time, json
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_EPOCHS,
    CHECKPOINT_PATH, RESULTS_PATH, SEED, NUM_CLASSES, VIOLENT_LABELS
)
from dataset import get_dataloaders
from model import HybridViolenceDetectorV2, HybridLossV2


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    p = (epoch - warmup) / max(total - warmup, 1)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    act_preds, act_labels = [], []
    scores, bin_labels    = [], []
    for x, cls_lbl, bin_lbl in loader:
        x       = x.to(device)
        cls_lbl = cls_lbl.to(device)
        bin_lbl = bin_lbl.to(device)
        logits, score = model(x)
        loss = criterion(logits, score, cls_lbl, bin_lbl)
        total_loss += loss.item()

        act_preds.extend(logits.argmax(1).cpu().numpy())
        act_labels.extend(cls_lbl.cpu().numpy())
        scores.extend(score.cpu().numpy())
        bin_labels.extend(bin_lbl.cpu().numpy())
    act_labels = np.array(act_labels)
    act_preds  = np.array(act_preds)
    scores     = np.array(scores)
    bin_labels = np.array(bin_labels)
    violent_set = set(VIOLENT_LABELS.keys())
    bin_preds   = np.array([1 if p in violent_set else 0 for p in act_preds])
    return {
        "loss":       total_loss / len(loader),
        "top1_acc":   accuracy_score(act_labels, act_preds),
        "anomaly_f1": f1_score(bin_labels, bin_preds, zero_division=0),
        "roc_auc":    roc_auc_score(bin_labels, scores),
    }


def train():
    set_seed(SEED)
    device = get_device()
    print(f"Устройство: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = HybridViolenceDetectorV2().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total:,} ({total/1e6:.2f}M)")
    criterion = HybridLossV2(alpha=0.7, beta=0.3, smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: lr_lambda(ep, WARMUP_EPOCHS, EPOCHS)
    )

    best_f1    = 0.0
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_top1_acc': [],
        'val_anomaly_f1': [],
        'val_roc_auc': []
    }
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ")
    print("="*60)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_loss = 0
        for x, cls_lbl, bin_lbl in train_loader:
            x       = x.to(device)
            cls_lbl = cls_lbl.to(device)
            bin_lbl = bin_lbl.to(device)

            optimizer.zero_grad()
            logits, score = model(x)
            loss = criterion(logits, score, cls_lbl, bin_lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        val_m = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        train_loss_avg = train_loss / len(train_loader)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_m['loss'])
        history['val_top1_acc'].append(val_m['top1_acc'])
        history['val_anomaly_f1'].append(val_m['anomaly_f1'])
        history['val_roc_auc'].append(val_m['roc_auc'])
        print(f"Эпоха {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss_avg:.4f}  "
              f"top1={val_m['top1_acc']*100:.1f}%  "
              f"f1={val_m['anomaly_f1']:.4f}  "
              f"auc={val_m['roc_auc']:.4f}  "
              f"lr={cur_lr:.2e}  ({elapsed:.0f}s)")
        if val_m["anomaly_f1"] > best_f1:
            best_f1 = val_m["anomaly_f1"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_f1": best_f1,
                "val_metrics": val_m,
            }, CHECKPOINT_PATH)
            print(f"Сохранён чекпоинт (F1={best_f1:.4f})")
    print("\n" + "="*60)
    print("ТЕСТ")
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"Top-1 Accuracy: {test_m['top1_acc']*100:.2f}%")
    print(f"Anomaly F1:     {test_m['anomaly_f1']:.4f}")
    print(f"ROC-AUC:        {test_m['roc_auc']:.4f}")
    history_path = os.path.join(RESULTS_PATH, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nИстория обучения сохранена в {history_path}")
    return model, test_m


if __name__ == "__main__":
    train()
