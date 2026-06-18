"""
Конфигурация: MS ST-GCN + Temporal Transformer + Action Recognition + Anomaly Score
Полный датасет NTU RGB+D 120 (все 120 классов)
"""

# ─── Датасет ───────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/ilvina/Desktop/dipl/ntu120_2d.pkl"

# Протокол разбиения: 'xsub' или 'xset'
SPLIT_PROTOCOL = 'xsub'

# Все 120 классов используются для action recognition
NUM_CLASSES = 120

# Классы противоправных действий (label = A_номер - 1)
VIOLENT_LABELS = {
    49:  "punching/slapping other person",
    50:  "kicking other person",
    51:  "pushing other person",
    56:  "touch other persons pocket",
    105: "hit other person with something",
    106: "wield knife towards other person",
    107: "knock over other person",
    108: "grab other persons stuff",
    109: "shoot at other person with a gun",
}

# ─── Скелет (COCO 17 точек) ────────────────────────────────────────────────────
NUM_JOINTS  = 17
IN_CHANNELS = 2

SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ─── Параметры модели ──────────────────────────────────────────────────────────
MAX_FRAMES  = 64
MAX_PERSONS = 2

# MS ST-GCN
STGCN_CHANNELS = [64, 128, 256]
STGCN_DROPOUT  = 0.3

# Temporal Transformer
TRANSFORMER_DIM     = 256
TRANSFORMER_HEADS   = 8
TRANSFORMER_LAYERS  = 4
TRANSFORMER_DROPOUT = 0.1

# ─── Обучение ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 36
LR           = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
SEED         = 42

# ─── Anomaly Score ─────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 0.5

# ─── Пути ──────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/Users/ilvina/Desktop/dipl/checkpoints/v2_best_model.pt"
RESULTS_PATH    = "/Users/ilvina/Desktop/dipl/results2/"
