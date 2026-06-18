import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report,
    roc_auc_score, f1_score, accuracy_score
)

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from config import (
    CHECKPOINT_PATH, RESULTS_PATH, VIOLENT_LABELS, NUM_CLASSES
)
from dataset import get_dataloaders
from model import HybridViolenceDetectorV2, HybridLossV2


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_logits, all_scores = [], []
    all_cls_labels, all_bin_labels = [], []

    for x, cls_lbl, bin_lbl in loader:
        x = x.to(device)
        logits, score = model(x)
        all_logits.append(logits.cpu())
        all_scores.append(score.cpu())
        all_cls_labels.append(cls_lbl)
        all_bin_labels.append(bin_lbl)
    logits     = torch.cat(all_logits).numpy()
    scores     = torch.cat(all_scores).numpy()
    cls_labels = torch.cat(all_cls_labels).numpy()
    bin_labels = torch.cat(all_bin_labels).numpy()
    cls_preds  = logits.argmax(axis=1)
    violent_set = set(VIOLENT_LABELS.keys())
    bin_preds   = np.array([1 if p in violent_set else 0 for p in cls_preds])

    return cls_preds, bin_preds, scores, cls_labels, bin_labels


def plot_roc(bin_labels, scores, save_path):
    fpr, tpr, _ = roc_curve(bin_labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#E74C3C", lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC-кривая", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    return roc_auc


def plot_confusion(bin_labels, bin_preds, save_path):
    cm = confusion_matrix(bin_labels, bin_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Нейтральное","Противоправное"])
    ax.set_yticklabels(["Нейтральное","Противоправное"])
    ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
    ax.set_title("Матрица ошибок", fontsize=13)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_anomaly_dist(bin_labels, scores, threshold, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores[bin_labels==0], bins=50, alpha=0.6,
            color="#2ECC71", label="Нейтральные", density=True)
    ax.hist(scores[bin_labels==1], bins=50, alpha=0.6,
            color="#E74C3C", label="Противоправные", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=2,
               label=f"Порог={threshold:.3f}")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Плотность")
    ax.set_title("Распределение Anomaly Score", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def plot_top_classes(cls_labels, cls_preds, save_path, top_n=20):
    errors = cls_labels[cls_labels != cls_preds]
    from collections import Counter
    top_err = Counter(errors).most_common(top_n)
    if not top_err: return

    class_names = {
        49:"punch", 50:"kick", 51:"push", 56:"pocket",
        105:"hit obj", 106:"knife", 107:"knock", 108:"grab", 109:"shoot"
    }

    labels_str = [class_names.get(c, f"A{c+1:03d}") for c, _ in top_err]
    counts     = [cnt for _, cnt in top_err]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C" if c in VIOLENT_LABELS else "#3498DB"
              for c, _ in top_err]
    ax.barh(labels_str, counts, color=colors)
    ax.set_xlabel("Число ошибок")
    ax.set_title(f"Топ-{top_n} классов по числу ошибок классификации", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


def full_evaluation():
    device = get_device()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    model = HybridViolenceDetectorV2().to(device)
    ck    = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    print(f"  Эпоха {ck['epoch']}, best F1={ck['best_f1']:.4f}")
    _, _, test_loader = get_dataloaders(batch_size=64)
    cls_preds, bin_preds, scores, cls_labels, bin_labels = \
        collect_predictions(model, test_loader, device)

    # порог
    fpr, tpr, thresholds = roc_curve(bin_labels, scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    bin_preds_opt = (scores >= best_thresh).astype(int)
    top1 = accuracy_score(cls_labels, cls_preds)
    print(f"Top-1 Accuracy: {top1*100:.2f}%")

    # топ 1 только на противоправных
    v_mask = np.isin(cls_labels, list(VIOLENT_LABELS.keys()))
    if v_mask.sum() > 0:
        top1_v = accuracy_score(cls_labels[v_mask], cls_preds[v_mask])
        print(f"Top-1 (только противоправные): {top1_v*100:.2f}%")
    print(classification_report(
        bin_labels, bin_preds_opt,
        target_names=["Нейтральное", "Противоправное"]
    ))
    roc_auc = roc_auc_score(bin_labels, scores)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Оптимальный порог: {best_thresh:.4f}")

    plot_roc(bin_labels, scores,
             os.path.join(RESULTS_PATH, "roc_curve.png"))
    plot_confusion(bin_labels, bin_preds_opt,
                   os.path.join(RESULTS_PATH, "confusion_matrix.png"))
    plot_anomaly_dist(bin_labels, scores, best_thresh,
                      os.path.join(RESULTS_PATH, "anomaly_score_dist.png"))
    plot_top_classes(cls_labels, cls_preds,
                     os.path.join(RESULTS_PATH, "top_errors.png"))


if name == "__main__":
    full_evaluation()