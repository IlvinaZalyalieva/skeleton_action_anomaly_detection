"""
Улучшенная гибридная архитектура v2:
  MS ST-GCN + Cross-Person Attention + Temporal Transformer + Dual Head

Новые компоненты по сравнению с v1:
  1. Dynamic Graph Convolution  — матрица смежности частично обучаемая
  2. Cross-Person Attention      — явное моделирование взаимодействия людей
  3. Multi-Scale Temporal Conv   — параллельные ядра 3/7/15 вместо одного 9
  4. Dual Head                   — action recognition (120 классов) + Anomaly Score
  5. Label Smoothing в потерях   — регуляризация для 120 классов

Схема:
  Вход (B, P, T, J, C)
      │
  [MS ST-GCN с Dynamic Graph]   ← пространственные признаки каждого человека
      │
  [Cross-Person Attention]       ← взаимодействие между людьми
      │
  [Multi-Scale Temporal Conv]    ← локальные временные паттерны
      │
  [Temporal Transformer]         ← глобальный временной контекст
      │
  [Dual Head]
      ├─ Action Head: logits (B, 120)   — какое действие
      └─ Anomaly Head: score (B,) [0,1] — насколько опасно
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_JOINTS, IN_CHANNELS, MAX_FRAMES, MAX_PERSONS, NUM_CLASSES,
    SKELETON_EDGES, STGCN_CHANNELS, STGCN_DROPOUT,
    TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS, TRANSFORMER_DROPOUT,
    VIOLENT_LABELS
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Граф скелета — статические матрицы смежности (3 масштаба)
# ══════════════════════════════════════════════════════════════════════════════

def build_adjacency_multiscale(num_joints: int, edges: list):
    J  = num_joints
    A1 = torch.zeros(J, J)
    for i, j in edges:
        A1[i, j] = 1.0
        A1[j, i] = 1.0
    A0     = torch.eye(J)
    A2_raw = A1 @ A1 - A1 - A0
    A2     = (A2_raw > 0).float()

    def norm(A):
        A = A + torch.eye(J) * 1e-3
        D = A.sum(1).clamp(min=1e-6)
        D_inv = torch.diag(D.pow(-0.5))
        return D_inv @ A @ D_inv

    return [norm(A0), norm(A1), norm(A2)]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dynamic Graph Conv (NEW)
#    Статическая матрица + обучаемая динамическая добавка
# ══════════════════════════════════════════════════════════════════════════════

class DynamicGraphConv(nn.Module):
    """
    Пространственная свёртка с динамической матрицей смежности.
    A_dynamic вычисляется из признаков через dot-product attention.
    Итоговая матрица: A = A_static * importance + A_dynamic * alpha

    Вход:  (B, C_in, T, J)
    Выход: (B, C_out, T, J)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 adjacency_list: list, alpha: float = 0.5):
        super().__init__()
        self.num_scales = len(adjacency_list)
        self.alpha      = nn.Parameter(torch.tensor(alpha))

        for i, A in enumerate(adjacency_list):
            self.register_buffer(f"A{i}", A)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.num_scales)
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(adjacency_list[0].shape))
            for _ in range(self.num_scales)
        ])

        # Для динамической матрицы: проекции Q и K
        self.q_proj = nn.Conv2d(in_channels, max(in_channels // 4, 16), kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, max(in_channels // 4, 16), kernel_size=1)
        self._qk_dim = max(in_channels // 4, 16)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C, T, J)
        B, C, T, J = x.shape

        # Динамическая матрица через attention по суставам
        Q = self.q_proj(x).mean(dim=2)  # (B, C//4, J)
        K = self.k_proj(x).mean(dim=2)  # (B, C//4, J)
        A_dyn = torch.softmax(
            torch.bmm(Q.transpose(1, 2), K) / math.sqrt(self._qk_dim),
            dim=-1
        )  # (B, J, J)

        out = 0
        for i in range(self.num_scales):
            A_static = getattr(self, f"A{i}") * self.edge_importance[i]
            # Комбинируем статическую и динамическую матрицы
            A_combined = A_static.unsqueeze(0) + self.alpha * A_dyn
            xi = torch.einsum("bctj,bjk->bctk", x, A_combined)
            xi = self.convs[i](xi)
            out = out + xi

        return self.bn(out)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Scale Temporal Conv (NEW)
#    Параллельные ядра разного размера → улавливает паттерны разной длины
# ══════════════════════════════════════════════════════════════════════════════

class MultiScaleTemporalConv(nn.Module):
    """
    Параллельные временные свёртки с ядрами 3, 7, 15 кадров.
    Результаты конкатенируются и проецируются обратно.

    Вход/Выход: (B, C, T, J)
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        branch_ch = channels // 4  # каналы на каждую ветку

        self.branches = nn.ModuleList()
        for k in [3, 7, 15]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, branch_ch,
                          kernel_size=(k, 1),
                          padding=((k-1)//2, 0),
                          stride=(stride, 1)),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            ))

        # Ветка с dilated conv для дальних зависимостей
        self.branches.append(nn.Sequential(
            nn.Conv2d(channels, branch_ch,
                      kernel_size=(3, 1),
                      padding=(2, 0),
                      dilation=(2, 1),
                      stride=(stride, 1)),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        ))

        # Проекция обратно в channels
        self.proj = nn.Sequential(
            nn.Conv2d(branch_ch * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

        # Residual для stride
        if stride != 1:
            self.res_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels)
            )
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        branches_out = [b(x) for b in self.branches]
        out = torch.cat(branches_out, dim=1)
        out = self.proj(out)
        return F.relu(out + self.res_conv(x))


# ══════════════════════════════════════════════════════════════════════════════
# 4. ST-GCN блок v2
# ══════════════════════════════════════════════════════════════════════════════

class STGCNBlockV2(nn.Module):
    """
    Блок ST-GCN v2:
      DynamicGraphConv → ReLU → MultiScaleTemporalConv → Dropout → Residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 adjacency_list: list, dropout: float = 0.3, stride: int = 1):
        super().__init__()
        self.dgcn = DynamicGraphConv(in_channels, out_channels, adjacency_list)
        self.mstcn = MultiScaleTemporalConv(out_channels, stride=stride)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x   = self.relu(self.dgcn(x))
        x   = self.drop(self.mstcn(x))
        return self.relu(x + res)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cross-Person Attention (NEW)
#    Явное моделирование взаимодействия между людьми
# ══════════════════════════════════════════════════════════════════════════════

class CrossPersonAttention(nn.Module):
    """
    Механизм внимания между признаками разных людей.
    Позволяет модели явно учитывать взаимодействие (удар, толчок, кража).

    Вход:  (B, P, T, C) — признаки P людей
    Выход: (B, P, T, C) — обогащённые признаки
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads,
            dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, P, T, C) → обрабатываем по каждому временному шагу
        B, P, T, C = x.shape
        # Reshape: (B*T, P, C) — внимание между людьми в каждый момент
        x_t = x.permute(0, 2, 1, 3).reshape(B * T, P, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = attn_out.reshape(B, T, P, C).permute(0, 2, 1, 3)
        return self.norm(x + attn_out)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Temporal Transformer
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        B   = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_enc(x)
        x   = self.encoder(x)
        return self.norm(x[:, 0])


# ══════════════════════════════════════════════════════════════════════════════
# 7. Dual Head: Action Recognition + Anomaly Score
# ══════════════════════════════════════════════════════════════════════════════

class DualHead(nn.Module):
    """
    Два выхода из общего признакового пространства:
      - action_logits : (B, 120) — распознавание конкретного действия
      - anomaly_score : (B,)     — степень аномальности [0, 1]

    Anomaly Score вычисляется двумя способами и комбинируется:
      1. Обучаемый нейрон (MSE к бинарной метке)
      2. Сумма вероятностей противоправных классов (soft mapping)
    """

    def __init__(self, in_features: int, num_classes: int,
                 violent_labels: list, dropout: float = 0.3):
        super().__init__()
        self.violent_labels = violent_labels

        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Action recognition head
        self.action_head = nn.Linear(256, num_classes)

        # Anomaly score head (обучаемый)
        self.score_head = nn.Linear(256, 1)

        # Веса для комбинирования двух источников score
        self.score_blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        feat = self.shared(x)

        # Action logits
        action_logits = self.action_head(feat)   # (B, 120)

        # Score 1: обучаемый нейрон
        score_learned = torch.sigmoid(self.score_head(feat)).squeeze(-1)  # (B,)

        # Score 2: сумма вероятностей противоправных классов
        probs = torch.softmax(action_logits, dim=-1)  # (B, 120)
        score_soft = probs[:, self.violent_labels].sum(dim=-1)  # (B,)

        # Комбинируем
        alpha = torch.sigmoid(self.score_blend)
        anomaly_score = alpha * score_learned + (1 - alpha) * score_soft

        return action_logits, anomaly_score


# ══════════════════════════════════════════════════════════════════════════════
# 8. Полная модель v2
# ══════════════════════════════════════════════════════════════════════════════

class HybridViolenceDetectorV2(nn.Module):
    """
    MS ST-GCN v2 + Cross-Person Attention + Temporal Transformer + Dual Head

    Вход:  (B, P, T, J, C)
    Выход: action_logits (B, 120), anomaly_score (B,)
    """

    def __init__(self):
        super().__init__()

        adj_list = build_adjacency_multiscale(NUM_JOINTS, SKELETON_EDGES)
        channels = [IN_CHANNELS] + STGCN_CHANNELS

        # ── MS ST-GCN блоки ──────────────────────────────────────────────────
        self.stgcn_blocks = nn.ModuleList()
        for i in range(len(STGCN_CHANNELS)):
            self.stgcn_blocks.append(
                STGCNBlockV2(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    adjacency_list=adj_list,
                    dropout=STGCN_DROPOUT,
                    stride=2 if i == 1 else 1
                )
            )

        stgcn_out = STGCN_CHANNELS[-1]  # 256

        # ── Cross-Person Attention ───────────────────────────────────────────
        self.cross_person_attn = CrossPersonAttention(dim=stgcn_out, heads=4)

        # ── Проекция в Transformer ───────────────────────────────────────────
        self.proj = nn.Linear(stgcn_out, TRANSFORMER_DIM)

        # ── Temporal Transformer ─────────────────────────────────────────────
        self.transformer = TemporalTransformer(
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            dropout=TRANSFORMER_DROPOUT
        )

        # ── Dual Head ────────────────────────────────────────────────────────
        violent_idx = sorted(VIOLENT_LABELS.keys())
        self.head = DualHead(
            in_features=TRANSFORMER_DIM,
            num_classes=NUM_CLASSES,
            violent_labels=violent_idx,
        )

    def forward(self, x):
        """x: (B, P, T, J, C)"""
        B, P, T, J, C = x.shape

        # ST-GCN для каждого человека
        person_feats = []
        for p in range(P):
            xp = x[:, p].permute(0, 3, 1, 2).contiguous()  # (B, C, T, J)
            for block in self.stgcn_blocks:
                xp = block(xp)
            xp = xp.mean(dim=-1).permute(0, 2, 1).contiguous()  # (B, T', C)
            person_feats.append(xp)

        # (B, P, T', C)
        persons = torch.stack(person_feats, dim=1)

        # Cross-Person Attention
        persons = self.cross_person_attn(persons)  # (B, P, T', C)

        # Агрегация по людям: max pooling
        x_agg = persons.max(dim=1).values          # (B, T', C)

        # Проекция + Transformer
        x_proj = self.proj(x_agg)                  # (B, T', D)
        x_cls  = self.transformer(x_proj)           # (B, D)

        # Dual Head
        action_logits, anomaly_score = self.head(x_cls)

        return action_logits, anomaly_score


# ══════════════════════════════════════════════════════════════════════════════
# 9. Функция потерь
# ══════════════════════════════════════════════════════════════════════════════

class HybridLossV2(nn.Module):
    """
    L = α * LabelSmoothingCE(action_logits, class_label)
      + β * MSE(anomaly_score, binary_label)

    LabelSmoothing помогает при 120 классах — предотвращает переуверенность.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 smoothing: float = 0.1,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.smoothing = smoothing
        self.ce        = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=smoothing
        )
        self.mse = nn.MSELoss()

    def forward(self, action_logits, anomaly_score, class_labels, binary_labels):
        loss_ce  = self.ce(action_logits, class_labels.long())
        loss_mse = self.mse(anomaly_score, binary_labels.float())
        return self.alpha * loss_ce + self.beta * loss_mse


# ══════════════════════════════════════════════════════════════════════════════
# Проверка
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    model = HybridViolenceDetectorV2()
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total:,} ({total/1e6:.2f}M)")

    dummy  = torch.randn(4, MAX_PERSONS, MAX_FRAMES, NUM_JOINTS, IN_CHANNELS)
    logits, score = model(dummy)
    print(f"action_logits: {logits.shape}")   # (4, 120)
    print(f"anomaly_score: {score.shape}")    # (4,)
    print(f"score range:   [{score.min():.3f}, {score.max():.3f}]")

    criterion = HybridLossV2()
    class_labels  = torch.randint(0, 120, (4,))
    binary_labels = torch.tensor([0, 1, 1, 0])
    loss = criterion(logits, score, class_labels, binary_labels)
    print(f"loss: {loss.item():.4f}")
