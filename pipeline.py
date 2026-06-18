import sys, os, time
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

sys.path.insert(0, '/Users/ilvina/Desktop/dipl')

from ultralytics import YOLO
from dataset import normalize_skeleton
from model import HybridViolenceDetectorV2
from config import VIOLENT_LABELS, CHECKPOINT_PATH
from inference import ACTION_NAMES

#VIDEO_DIR  = "/Users/ilvina/Desktop/dipl/nturgb+d_rgb"
#VIDEO_NAME = "S032C002P102R002A107_rgb.avi"   

NUM_JOINTS   = 17
MAX_FRAMES   = 64
MAX_PERSONS  = 2
THRESHOLD    = 0.45      
WINDOW_SIZE  = MAX_FRAMES 
STEP         = 16         

def get_device():
    if torch.cuda.is_available():           return torch.device('cuda')
    if torch.backends.mps.is_available():   return torch.device('mps')
    return torch.device('cpu')


def extract_skeletons_from_video(video_path: str, yolo_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Видео: {os.path.basename(video_path)}")
    print(f"  Разрешение: {w}×{h}, FPS: {fps_video:.1f}, Кадров: {total_frames}")
    all_frames_skeletons = []  
    all_frames_images    = []  

    t0 = time.perf_counter()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, verbose=False)
        result  = results[0]
        frame_persons = [] 
        if result.keypoints is not None and len(result.keypoints) > 0:
            for i in range(min(len(result.keypoints), MAX_PERSONS)):
                kp_xy   = result.keypoints.xy[i].cpu().numpy()    # (17, 2)
                kp_conf = result.keypoints.conf[i].cpu().numpy()  # (17,)
                frame_persons.append((kp_xy, kp_conf))

        all_frames_skeletons.append(frame_persons)
        all_frames_images.append(frame.copy())
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f"  Извлечено кадров: {frame_idx} за {elapsed:.1f}с "
          f"({frame_idx/elapsed:.1f} FPS)")

    return all_frames_skeletons, all_frames_images, fps_video

def build_skeleton_sequence(frames_window):
    T = len(frames_window)
    kp_seq    = np.zeros((MAX_PERSONS, T, NUM_JOINTS, 2), dtype=np.float32)
    conf_seq  = np.zeros((MAX_PERSONS, T, NUM_JOINTS),    dtype=np.float32)
    for t, frame_persons in enumerate(frames_window):
        for p, (kp_xy, kp_conf) in enumerate(frame_persons[:MAX_PERSONS]):
            kp_seq[p, t]   = kp_xy
            conf_seq[p, t] = kp_conf
    kp_seq = kp_seq * conf_seq[:, :, :, np.newaxis]
    kp_seq = normalize_skeleton(kp_seq)
    if T >= MAX_FRAMES:
        idx = np.linspace(0, T - 1, MAX_FRAMES, dtype=int)
        kp_seq = kp_seq[:, idx]
    else:
        pad = np.zeros((MAX_PERSONS, MAX_FRAMES - T, NUM_JOINTS, 2), dtype=np.float32)
        kp_seq = np.concatenate([kp_seq, pad], axis=1)

    return kp_seq  # (MAX_PERSONS, MAX_FRAMES, 17, 2)

@torch.no_grad()
def classify_window(kp_seq, model, device):
    x = torch.from_numpy(kp_seq).unsqueeze(0).to(device)  # (1, P, T, J, C)
    logits, score = model(x)
    probs     = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_cls  = int(probs.argmax())
    anom_score = float(score[0].cpu())
    is_violent = anom_score >= THRESHOLD
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(ACTION_NAMES[i], round(float(probs[i]), 3)) for i in top3_idx]
    return pred_cls, ACTION_NAMES[pred_cls], anom_score, is_violent, top3


def run_pipeline(video_path: str, save_video: str = None):
    device = get_device()
    print(f"Устройство: {device}\n")
    yolo = YOLO('yolo11n-pose.pt')
    our_model = HybridViolenceDetectorV2().to(device)
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    our_model.load_state_dict(ck['model_state'])
    our_model.eval()
    print("=" * 60)
    print("Извлечение скелетов")
    print("=" * 60)
    all_skeletons, all_frames, fps_video = \
        extract_skeletons_from_video(video_path, yolo)

    total_frames = len(all_skeletons)
    persons_per_frame = [len(f) for f in all_skeletons]
    print(f"  Среднее людей в кадре: {np.mean(persons_per_frame):.1f}")
    print(f"  Кадров с ≥2 людьми: "
          f"{sum(1 for p in persons_per_frame if p >= 2)}/{total_frames}")

    print("\n" + "=" * 60)
    print("Выход модели")
    print("=" * 60)

    window_results = []
    t_inference = []
    if total_frames < WINDOW_SIZE:
        windows = [(0, total_frames)]
    else:
        windows = [(s, s + WINDOW_SIZE)
                   for s in range(0, total_frames - WINDOW_SIZE + 1, STEP)]

    for start, end in windows:
        window = all_skeletons[start:end]
        kp_seq = build_skeleton_sequence(window)
        t0 = time.perf_counter()
        pred_cls, action_name, anom_score, is_violent, top3 = \
            classify_window(kp_seq, our_model, device)
        t1 = time.perf_counter()
        t_inference.append(t1 - t0)

        window_results.append({
            'start_frame':  start,
            'end_frame':    end,
            'start_sec':    round(start / fps_video, 2),
            'end_sec':      round(end   / fps_video, 2),
            'action_class': pred_cls,
            'action_name':  action_name,
            'anomaly_score':round(anom_score, 4),
            'is_violent':   is_violent,
            'top3':         top3,
        })

    print(f"\nОбработано окон: {len(window_results)}")
    print(f"Среднее время инференса на окно: {np.mean(t_inference)*1000:.1f} мс")
    print(f"FPS пайплайна (полный): "
          f"{total_frames / (total_frames/fps_video + sum(t_inference)):.1f}")

    violent_windows = [r for r in window_results if r['is_violent']]
    print(f"\nОбнаружено противоправных окон: {len(violent_windows)} / {len(window_results)}")
    for r in window_results:
        flag = "ОПАСНО" if r['is_violent'] else "   норма "
        print(f"\n[{r['start_sec']:5.1f}s – {r['end_sec']:5.1f}s]  "
              f"{flag}  score={r['anomaly_score']:.3f}")
        print(f"     Топ-3 предсказанных действий:")
        for name, prob in r['top3']:
            v_mark = "!" if any(name == ACTION_NAMES[k]
                                   for k in VIOLENT_LABELS) else ""
            print(f"       {name[:40]:<40} {prob*100:.1f}%{v_mark}")

    return window_results


if __name__ == '__main__':
    video_path = sys.argv[1]
    VIDEO_NAME = video_path.split('/')[-1]
    action_num = int(VIDEO_NAME.split('A')[1][:3])  # A107 → 107
    true_label = action_num - 1                      
    true_name  = ACTION_NAMES[true_label] if true_label < len(ACTION_NAMES) else f"A{action_num}"
    true_type  = "ПРОТИВОПРАВНОЕ" if true_label in VIOLENT_LABELS else "нейтральное"
    print(f"Видео:       {VIDEO_NAME}")
    print(f"Истина:      {true_type} — {true_name}")
    print()
    run_pipeline(video_path)
