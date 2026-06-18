import numpy as np
import torch
from config import CHECKPOINT_PATH, VIOLENT_LABELS, NUM_CLASSES
from dataset import preprocess
from model import HybridViolenceDetectorV2
import pickle
from config import DATA_PATH

ACTION_NAMES = [
    "drink water","eat meal","brushing teeth","brushing hair","drop",
    "pickup","throw","sitting down","standing up","clapping",
    "reading","writing","tear up paper","wear jacket","take off jacket",
    "wear a shoe","take off a shoe","wear on glasses","take off glasses","put on hat",
    "take off hat","cheer up","hand waving","kicking something","reach into pocket",
    "hopping","jump up","make a phone call","playing with phone","typing on keyboard",
    "pointing to something","taking a selfie","check time","rub two hands","nod head",
    "shake head","wipe face","salute","put palms together","cross hands (stop)",
    "sneeze/cough","staggering","falling","touch head","touch chest",
    "touch back","touch neck","nausea/vomiting","use a fan","punching/slapping",
    "kicking other person","pushing other person","pat on back","point finger at other",
    "hugging","giving something","touch others pocket","handshaking","walking towards",
    "walking apart","put on headphone","take off headphone","shoot at basket",
    "bounce ball","tennis bat swing","juggling table tennis","hush","flick hair",
    "thumb up","thumb down","make ok sign","make victory sign","staple book",
    "counting money","cutting nails","cutting paper","snapping fingers","open bottle",
    "sniff","squat down","toss a coin","fold paper","ball up paper",
    "play magic cube","apply cream on face","apply cream on hand","put on bag","take off bag",
    "put something into bag","take something out of bag","open a box","move heavy objects","shake fist",
    "throw up cap","hands up","cross arms","arm circles","arm swings",
    "running on spot","butt kicks","cross toe touch","side kick","yawn",
    "stretch oneself","blow nose","hit other with something","wield knife","knock over other",
    "grab others stuff","shoot at other with gun","step on foot","high-five","cheers and drink",
    "carry something with other","take photo of other","follow other person","whisper in ear","exchange things",
    "support somebody","finger-guessing game"
]

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(path=CHECKPOINT_PATH):
    device = get_device()
    model  = HybridViolenceDetectorV2().to(device)
    ck     = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"Модель загружена (эпоха {ck['epoch']}, F1={ck['best_f1']:.4f})")
    return model, device


@torch.no_grad()
def predict(model, device, keypoints: np.ndarray,
            kp_scores: np.ndarray = None,
            threshold: float = 0.5):
    if keypoints.ndim == 3:
        keypoints = keypoints[np.newaxis]
        if kp_scores is not None:
            kp_scores = kp_scores[np.newaxis]
    ann = {
        "keypoint":       keypoints.astype(np.float32),
        "keypoint_score": kp_scores.astype(np.float32) if kp_scores is not None
                          else np.ones(keypoints.shape[:3], dtype=np.float32),
        "total_frames":   keypoints.shape[1],
        "label":          0,
    }
    kp = preprocess(ann)
    x  = torch.from_numpy(kp).unsqueeze(0).to(device)  # (1, P, T, J, C)
    logits, score = model(x)
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_class    = int(probs.argmax())
    anomaly_score = float(score[0].cpu())
    is_violent    = pred_class in VIOLENT_LABELS or anomaly_score >= threshold
    top5_idx = probs.argsort()[::-1][:5]
    top5 = [(ACTION_NAMES[i], round(float(probs[i]), 4)) for i in top5_idx]

    return {
        "action_class":  pred_class,
        "action_name":   ACTION_NAMES[pred_class],
        "anomaly_score": round(anomaly_score, 4),
        "is_violent":    is_violent,
        "top5":          top5,
    }


def demo():
    with open(DATA_PATH, "rb") as f:
        raw = pickle.load(f)
    ann_list = raw["annotations"]
    model, device = load_model()
    violent_samples = [a for a in ann_list if a["label"] in VIOLENT_LABELS][:4]
    neutral_samples = [a for a in ann_list if a["label"] not in VIOLENT_LABELS][:4]
    print("\n" + "="*65)
    print("ДЕМО ИНФЕРЕНСА")
    print("="*65)
    for ann in violent_samples + neutral_samples:
        kp    = ann["keypoint"]
        sc    = ann["keypoint_score"]
        true  = ann["label"]
        true_name = ACTION_NAMES[true]
        true_type = "ПРОТИВОПРАВНОЕ" if true in VIOLENT_LABELS else "нейтральное"

        result = predict(model, device, kp, sc)
        ok = "!" if result["is_violent"] == (true in VIOLENT_LABELS) else " "

        print(f"\n{ok} {ann['frame_dir']}")
        print(f"  Истина:        {true_type} ({true_name})")
        print(f"  Предсказание:  {result['action_name']}")
        print(f"  Anomaly Score: {result['anomaly_score']}")
        print(f"  Опасно:        {'ДА' if result['is_violent'] else 'нет'}")
        print(f"  Топ-3: {result['top5'][:3]}")


if __name__ == "__main__":
    demo()
