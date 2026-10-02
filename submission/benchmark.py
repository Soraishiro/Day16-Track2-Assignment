#!/usr/bin/env python3
"""Benchmark LightGBM tren Credit Card Fraud Detection dataset.

Do lai: load data, training, chat luong (AUC/Accuracy/F1/Precision/Recall),
inference latency 1 dong va throughput 1000 dong. Ket qua ghi ra
benchmark_result.json va in ra terminal.

Chay:  python3 benchmark.py
"""

import inspect
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Tham so. Giu sat code goi y cua runbook de khi bao cao thi doi chieu duoc.
# --------------------------------------------------------------------------
SEED = 16
DATA_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")
TARGET = "Class"

# Tach 60 / 20 / 20.
#   lan 1: test_size=0.2   -> 20% test, con 80% la trainval
#   lan 2: test_size=0.25  -> 25% CUA 80% = 20% tong -> validation 20%, train 60%
# Vì sao lan hai la 0.25 chứ không phai 0.2: 0.2 x 0.8 = 0.16 (train 64/val 16),
# khong phai ti le 60/20/20. Muon validation cung 20% voi test thi phai tach
# 0.25/0.75 cua phan 80% con lai.
TEST_SIZE = 0.2
VAL_SIZE = 0.25

N_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 20
LEARNING_RATE = 0.05

# So thread cho LightGBM. Tai lieu: dat bang so CPU core that, khong phai so
# thread (CPU thuong dung hyper-threading). Khong dat qua lon voi dataset nho.
NUM_THREADS = os.cpu_count() or 1

# Can bang lop: chi dung MOT trong hai. None = de mac dinh, khong can bang.
# Tai lieu canh bao ca hai tham so deu lam xauu uoc luong xac suat tung lop, nen
# mac dinh de tat; chi bat khi uu tien Recall.
SCALE_POS_WEIGHT = None

# Nguong quyet dinh khac 0.5: mac dinh cua predict la 0.5, nhung chon bang
# validation moi dung chuan. KHONG chon tren tap test.
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID_MIN = 0.05
THRESHOLD_GRID_MAX = 0.95
THRESHOLD_GRID_STEP = 0.01
# Tieu chi chon nguong. F1 la tieu chi can bang giua precision va recall.
# Trong thuc te ngan hang, chi phi bo sot mot vu gian lan thuong lon hon chi phi
# bao dong gia, nen can van dung trong so khi chon nguong.
THRESHOLD_CRITERION = "f1"

# So lan lap do latency. Lan dau tien bo qua (warm-up) de khong do thoi gian
# khoi tao lai bo dem / CPU thang tanh.
LATENCY_REPEATS = 50
LATENCY_WARMUP = 5
BATCH_ROWS = 1000
BATCH_REPEATS = 10
BATCH_WARMUP = 3


# --------------------------------------------------------------------------
# Chon nguong quyet dinh
# --------------------------------------------------------------------------
def select_threshold(y_true, probabilities, criterion: str = "f1") -> tuple:
    """Chon nguong phan loai tren TAP VALIDATION.

    Tra ve (nguong, f1_tai_nguong, bang tong hop cac moc da kiem tra).

    Dung validation chu khong dung test: neu chon nguong tren test thi con so
    AUC/F1/FN cuoi cung se khong con la so do tren du lieu chua thay.
    """
    grid = np.arange(THRESHOLD_GRID_MIN, THRESHOLD_GRID_MAX, THRESHOLD_GRID_STEP)
    rows = []
    for t in grid:
        hard = (probabilities >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 4),
                "precision": precision_score(y_true, hard, zero_division=0),
                "recall": recall_score(y_true, hard, zero_division=0),
                "f1": f1_score(y_true, hard, zero_division=0),
                "flagged": int(hard.sum()),
            }
        )
    best = max(rows, key=lambda r: r[criterion])
    return best["threshold"], best[criterion], rows


# --------------------------------------------------------------------------
# Mo ta moi truong do
# --------------------------------------------------------------------------
def describe_runtime() -> dict:
    """Thu thong tin may chay de gan vao bao cao.

    Cac truong AWS lay tu IMDSv2. Neu khong goi duoc (khong phai EC2, hoac IMDS
    tat) thi de trong, khong lam script fail.
    """
    info = {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "lightgbm_version": lgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "num_threads_used": NUM_THREADS,
    }

    token = None
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            token = resp.read().decode()
    except Exception:
        pass

    for key, path in (
        ("instance_type", "instance-type"),
        # IMDSv2 khong co path "region"; phai dung "placement/region".
        ("region", "placement/region"),
    ):
        try:
            req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}")
            if token:
                req.add_header("X-aws-ec2-metadata-token", token)
            with urllib.request.urlopen(req, timeout=1) as resp:
                info[key] = resp.read().decode()
        except Exception:
            info[key] = None

    return info


# --------------------------------------------------------------------------
# Do latency / throughput
# --------------------------------------------------------------------------
def measure_seconds(predict, data, repeats: int) -> tuple:
    """Chay `predict(data)` `repeats` lan, tra ve (median giay, trung binh giay, nhanh nhat giay)."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(data)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return (
        samples[len(samples) // 2],
        sum(samples) / len(samples),
        samples[0],
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Khong tim thay {DATA_PATH}. Chay o thu muc chua creditcard.csv "
            "(vi du ~/ml-benchmark) hoac tai dataset truoc."
        )

    runtime = describe_runtime()
    print("=" * 72)
    print("LightGBM benchmark - Credit Card Fraud Detection")
    print("=" * 72)
    print(f"Thoi diem do (UTC)   : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"May                 : {runtime.get('instance_type') or runtime['hostname']}")
    print(f"Region              : {runtime.get('region') or 'khong xac dinh'}")
    print(f"Kien truc           : {runtime['architecture']}")
    print(f"CPU / thread        : {runtime['cpu_count']} core / {NUM_THREADS} thread LightGBM")
    print(f"lightgbm {lgb.__version__} | sklearn {sklearn.__version__} | "
          f"pandas {pd.__version__} | numpy {np.__version__}")
    print("-" * 72)

    # 1. Load data ---------------------------------------------------------
    started = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    data_load_seconds = time.perf_counter() - started
    print(f"[1] Load data            : {data_load_seconds:.3f}s  shape={df.shape}")
    print(f"    Missing values       : {int(df.isna().sum().sum())}")
    print(f"    Class distribution   : {df[TARGET].value_counts().to_dict()}")

    # 2. Tach 60 / 20 / 20 -------------------------------------------------
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED, stratify=y_trainval
    )
    print(f"[2] Split                : train={len(X_train)}  "
          f"valid={len(X_valid)}  test={len(X_test)}")
    print(f"    Test chi dung danh gia cuoi. Early stopping dung tap valid.")

    # 3. Training ----------------------------------------------------------
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        random_state=SEED,
        n_jobs=NUM_THREADS,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="auc",
        # deterministic + force_col_wise: theo tai lieu, dat deterministic=true
        # ma khong dat force_col_wise/force_row_wise thi ket qua co so co the
        # khong on dinh. Trade la cham mot chut.
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

    # LightGBM 4.7 deprecated eval_set, thay bang eval_X/eval_y. Dung introspect
    # de chay duoc tren ca ban moi va ban cu, tranh canh bao deprecation.
    if "eval_X" in inspect.signature(model.fit).parameters:
        eval_kwargs = {"eval_X": X_valid, "eval_y": y_valid}
    else:
        eval_kwargs = {"eval_set": [(X_valid, y_valid)]}

    print(f"[3] Training (n_estimators={N_ESTIMATORS}, early_stopping={EARLY_STOPPING_ROUNDS})...")
    started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=50),
        ],
        **eval_kwargs,
    )
    training_seconds = time.perf_counter() - started
    best_iteration = model.best_iteration_ if model.best_iteration_ else N_ESTIMATORS
    print(f"    Training xong        : {training_seconds:.3f}s  best_iteration={best_iteration}")

    # 4. Chon nguong tren VALIDATION, roi moi danh gia tren tap test --------
    val_probabilities = model.predict_proba(X_valid)[:, 1]
    threshold, val_f1_at_threshold, threshold_scan = select_threshold(
        y_valid, val_probabilities, THRESHOLD_CRITERION
    )
    print(f"[4] Nguong (ch tren validation, toi uu {THRESHOLD_CRITERION}) = {threshold:.2f}"
          f"   F1_validation = {val_f1_at_threshold:.4f}")

    # AUC dung xac suat; cac chi do con lai dung nhan du doan.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"[5] Test metrics         : AUC={auc_roc:.6f} ACC={accuracy:.6f} "
          f"F1={f1:.6f} P={precision:.6f} R={recall:.6f}")
    print(f"    Confusion matrix     : TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"    -> bo sot {fn} vụ gian lan tren tong {int((y_test == 1).sum())} vụ trong tap test")

    # 5. Latency 1 dong va throughput 1000 dong ----------------------------
    # Do hai lan:
    #   (a) qua sklearn predict_proba tren DataFrame - gom ca chi phi pandas,
    #       dung khi pipeline thuc su nhan DataFrame. Day la so chinh.
    #   (b) qua booster_.predict tren ndarray ndarray - chi thoi gian cua LightGBM.
    #       Dung booster_ de khong kich hoat kiem tra ten cot cua sklearn.
    one_row_df, batch_df = X_test.iloc[:1], X_test.iloc[:BATCH_ROWS]
    one_row_np = one_row_df.to_numpy(dtype=np.float32)
    batch_np = np.ascontiguousarray(batch_df.to_numpy(dtype=np.float32))

    print(f"[6] Do latency ({LATENCY_REPEATS} lan) va throughput (batch {BATCH_ROWS} dong)...")

    for _ in range(LATENCY_WARMUP):
        model.predict_proba(one_row_df)
    single_df_s, single_df_avg, single_df_min = measure_seconds(
        model.predict_proba, one_row_df, LATENCY_REPEATS
    )

    for _ in range(BATCH_WARMUP):
        model.predict_proba(batch_df)
    batch_df_s, batch_df_avg, batch_df_min = measure_seconds(
        model.predict_proba, batch_df, BATCH_REPEATS
    )

    for _ in range(LATENCY_WARMUP):
        model.booster_.predict(one_row_np)
    single_np_s, single_np_avg, single_np_min = measure_seconds(
        model.booster_.predict, one_row_np, LATENCY_REPEATS
    )

    for _ in range(BATCH_WARMUP):
        model.booster_.predict(batch_np)
    batch_np_s, batch_np_avg, batch_np_min = measure_seconds(
        model.booster_.predict, batch_np, BATCH_REPEATS
    )

    # 6. Luu ket qua -------------------------------------------------------
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": runtime["architecture"],
        "instance_type": runtime.get("instance_type"),
        "region": runtime.get("region"),
        "versions": {
            "python": runtime["python_version"],
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_rows": len(df),
        "dataset_columns": int(df.shape[1]),
        "fraud_rows": int(y.sum()),
        "fraud_ratio": round(float(y.mean()), 6),
        "seed": SEED,
        "split": {
            "train": len(X_train),
            "validation": len(X_valid),
            "test": len(X_test),
            "test_size": TEST_SIZE,
            "validation_size_of_trainval": VAL_SIZE,
            "stratified": True,
            "note": (
                "60/20/20. Lan 2 tach 0.25 cua 80% con lai de validation = 20% tong. "
                "Tap test chi dung danh gia cuoi, khong dung chon so vong."
            ),
        },
        "n_jobs": NUM_THREADS,
        "decision_threshold": threshold,
        "threshold_selection": {
            "value": threshold,
            "criterion": THRESHOLD_CRITERION,
            "selected_on": "validation",
            "grid": {
                "min": THRESHOLD_GRID_MIN,
                "max": THRESHOLD_GRID_MAX,
                "step": THRESHOLD_GRID_STEP,
            },
            "sklearn_default_threshold": DEFAULT_THRESHOLD,
            "f1_on_validation_at_threshold": float(val_f1_at_threshold),
            "auc_on_validation": float(roc_auc_score(y_valid, val_probabilities)),
            "note": (
                "Nguong chon tren tap validation. Tap test chi dung mot lan de danh gia, "
                "khong dung de chon nguong hay chon so vong."
            ),
            "scan": threshold_scan,
        },
        "confusion_matrix_at_threshold": {
            "labels": [0, 1],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "rows_are_actual_cols_are_predicted": cm.tolist(),
        },
        "model": {
            "estimator": "LGBMClassifier",
            "n_estimators": N_ESTIMATORS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "learning_rate": LEARNING_RATE,
            "num_leaves": 31,
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "deterministic": True,
        },
        "data_load_seconds": round(data_load_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": float(auc_roc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "latency_1_row_ms": single_df_s * 1000,
        "latency_repeats": LATENCY_REPEATS,
        "latency_warmup": LATENCY_WARMUP,
        "latency_unit": "milliseconds",
        "latency_statistic": "median",
        "batch_rows": len(batch_df),
        "batch_repeats": BATCH_REPEATS,
        "batch_1000_rows_seconds": batch_df_s,
        "throughput_1000_rows_per_second": len(batch_df) / batch_df_s,
        "throughput_unit": "rows_per_second",
        "latency_1_row_ms_numpy_only": single_np_s * 1000,
        "throughput_1000_rows_per_second_numpy_only": len(batch_np) / batch_np_s,
        "timing_detail": {
            "sklearn_pandas": {
                "latency_1_row_ms_median": single_df_s * 1000,
                "latency_1_row_ms_mean": single_df_avg * 1000,
                "latency_1_row_ms_min": single_df_min * 1000,
                "batch_seconds_median": batch_df_s,
                "batch_seconds_mean": batch_df_avg,
                "note": "model.predict_proba(DataFrame) - gom ca chi phi pandas/sklearn.",
            },
            "booster_numpy": {
                "latency_1_row_ms_median": single_np_s * 1000,
                "latency_1_row_ms_mean": single_np_avg * 1000,
                "latency_1_row_ms_min": single_np_min * 1000,
                "batch_seconds_median": batch_np_s,
                "batch_seconds_mean": batch_np_avg,
                "note": "model.booster_.predict(ndarray) - chi thoi gian cua LightGBM.",
            },
        },
        "timing_summary": (
            "median; warm-up excluded; pho chinh do tren pandas DataFrame "
            "(predict_proba), gia tri *_numpy_only do tren ndarray thuan."
        ),
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # 7. In bang tong ket (khop bang trong README) --------------------------
    print("=" * 72)
    print("KET QUA")
    print("=" * 72)
    print(f"{'Metric':<40}{'Ket qua':>26}")
    print("-" * 72)
    print(f"{'Thoi gian load data':<40}{result['data_load_seconds']:>20.4f} s")
    print(f"{'Thoi gian training':<40}{result['training_seconds']:>20.4f} s")
    print(f"{'Best iteration':<40}{result['best_iteration']:>26}")
    print(f"{'AUC-ROC':<40}{result['auc_roc']:>26.6f}")
    print(f"{'Accuracy':<40}{result['accuracy']:>26.6f}")
    print(f"{'F1-Score':<40}{result['f1']:>26.6f}")
    print(f"{'Precision':<40}{result['precision']:>26.6f}")
    print(f"{'Recall':<40}{result['recall']:>26.6f}")
    print(f"{'  (nguong quyet dinh, chon tren validation)':<40}{result['decision_threshold']:>26.2f}")
    print(f"{'Confusion matrix TN/FP/FN/TP':<40}"
          f"{result['confusion_matrix_at_threshold']['tn']:>8,}"
          f"{result['confusion_matrix_at_threshold']['fp']:>6,}"
          f"{result['confusion_matrix_at_threshold']['fn']:>5,}"
          f"{result['confusion_matrix_at_threshold']['tp']:>5,}")
    print(f"{'Inference latency (1 row)':<40}{result['latency_1_row_ms']:>20.4f} ms")
    print(f"{'Inference throughput (1000 rows)':<40}"
          f"{result['throughput_1000_rows_per_second']:>18.1f} dong/s")
    print("-" * 72)
    print(f"{'(latency/throughput chi LightGBM, bo pandas)':<40}"
          f"{result['latency_1_row_ms_numpy_only']:>20.4f} ms  1 row")
    print(f"{'':40}{result['throughput_1000_rows_per_second_numpy_only']:>18.1f} dong/s  1000 rows")
    print("-" * 72)
    print(f"Da ghi ket qua: {RESULT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
