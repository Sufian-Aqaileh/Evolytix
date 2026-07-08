import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
import time, uuid, dataclasses
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

@dataclass
class FeedbackRecord:
    record_id: str
    model_output: Any
    feedback: Any
    metadata: dict
    timestamp: float = field(default_factory=time.time)


class MemoryBuffer:

    def __init__(self, max_size: int = 500):
        self._buf: deque = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, record: FeedbackRecord) -> None:
        self._buf.append(record)

    def all(self) -> List[FeedbackRecord]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def stats(self) -> dict:
        return {"size": len(self._buf), "max_size": self.max_size}


FeedbackValidator = Callable[[Any, Any], bool]


class FeedbackLogger:
    def __init__(
        self,
        buffer: MemoryBuffer,
        validator: Optional[FeedbackValidator] = None,
        on_record: Optional[Callable[[FeedbackRecord], None]] = None,
    ):
        self.buffer = buffer
        self.validator = validator
        self.on_record = on_record

    def log(
        self,
        model_output: Any,
        feedback: Any = None,
        metadata: Optional[dict] = None,
    ) -> FeedbackRecord:
        if self.validator and not self.validator(model_output, feedback):
            raise ValueError("FeedbackValidator rejected the record.")
        record = FeedbackRecord(
            record_id=str(uuid.uuid4()),
            model_output=model_output,
            feedback=feedback,
            metadata=metadata or {},
        )
        self.buffer.add(record)
        if self.on_record:
            self.on_record(record)
        return record


@dataclass
class DriftResult:
    drift_detected: bool
    score: float
    reason: str


class MeanShiftDetector:
    def __init__(
        self,
        reference_mean: float = 0.5,
        reference_std: float = 0.1,
        z_threshold: float = 2.5,
    ):
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.z_threshold = z_threshold

    def check(self, records: List[FeedbackRecord]) -> DriftResult:
        if not records:
            return DriftResult(False, 0.0, "no records")
        values = [r.model_output for r in records if isinstance(r.model_output, (int, float))]
        if not values:
            return DriftResult(False, 0.0, "no numeric outputs")
        current_mean = float(np.mean(values))
        z = abs(current_mean - self.reference_mean) / max(self.reference_std, 1e-9)
        detected = z > self.z_threshold
        return DriftResult(
            drift_detected=detected,
            score=z,
            reason=f"z={z:.3f} (threshold={self.z_threshold})",
        )


class DriftDetector:
    def __init__(self, strategies: Optional[List[MeanShiftDetector]] = None):
        self.strategies = strategies or [MeanShiftDetector()]

    def check(self, records: List[FeedbackRecord]) -> DriftResult:
        best = DriftResult(False, 0.0, "no strategies")
        for s in self.strategies:
            r = s.check(records)
            if r.score > best.score:
                best = r
        return best


@dataclass
class AdaptationAction:
    name: str
    handler: Callable[[DriftResult], None]


@dataclass
class AdaptationEvent:
    timestamp: float
    drift_result: DriftResult
    actions_taken: List[str]
    new_threshold: float


class AdaptationEngine:
    def __init__(
        self,
        buffer: MemoryBuffer,
        threshold_updater: Optional[Callable] = None,
        retrain_handler: Optional[Callable] = None,
        actions: Optional[List[AdaptationAction]] = None,
        min_interval_seconds: float = 0.0,
        initial_threshold: float = 3.0,
    ):
        self.buffer = buffer
        self.threshold_updater = threshold_updater
        self.retrain_handler = retrain_handler
        self.actions = actions or []
        self.min_interval_seconds = min_interval_seconds
        self.current_threshold = initial_threshold
        self.history: List[AdaptationEvent] = []
        self._last_adaptation_time: float = 0.0

    def adapt(self, drift_result: DriftResult) -> Optional[AdaptationEvent]:
        if not drift_result.drift_detected:
            return None
        now = time.time()
        if now - self._last_adaptation_time < self.min_interval_seconds:
            return None
        self._last_adaptation_time = now
        taken: List[str] = []
        # Built-in: call threshold_updater if provided
        if self.threshold_updater:
            self.current_threshold = self.threshold_updater(
                drift_result, self.current_threshold
            )
            taken.append("threshold_update")
        # Built-in: call retrain_handler if provided
        if self.retrain_handler:
            self.retrain_handler(self.buffer.all())
            taken.append("retrain")
        # Custom actions
        for action in self.actions:
            action.handler(drift_result)
            taken.append(action.name)
        event = AdaptationEvent(
            timestamp=now,
            drift_result=drift_result,
            actions_taken=taken,
            new_threshold=self.current_threshold,
        )
        self.history.append(event)
        return event


class AdaptiveLayer:
    def __init__(
        self,
        buffer: MemoryBuffer,
        logger: FeedbackLogger,
        detector: DriftDetector,
        engine: AdaptationEngine,
        check_every: int = 10,
    ):
        self.buffer = buffer
        self.logger = logger
        self.detector = detector
        self.engine = engine
        self.check_every = max(1, check_every)
        self._records_since_check: int = 0

    def process(
        self,
        model_output: Any,
        feedback: Any = None,
        metadata: Optional[dict] = None,
        force_check: bool = False,
    ) -> Optional[AdaptationEvent]:
        self.logger.log(model_output, feedback, metadata)
        self._records_since_check += 1
        if not force_check and self._records_since_check < self.check_every:
            return None
        self._records_since_check = 0
        drift_result = self.detector.check(self.buffer.all())
        return self.engine.adapt(drift_result)

    def process_batch(
        self,
        outputs: List[Any],
        feedbacks: Optional[List[Any]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> List[Optional[AdaptationEvent]]:
        feedbacks = feedbacks or [None] * len(outputs)
        metadata = metadata or [{}] * len(outputs)
        return [self.process(o, f, m) for o, f, m in zip(outputs, feedbacks, metadata)]

    @property
    def current_threshold(self) -> float:
        return self.engine.current_threshold

    def status(self) -> dict:
        return {
            "buffer": self.buffer.stats(),
            "current_threshold": self.engine.current_threshold,
            "adaptation_events": len(self.engine.history),
            "records_since_last_check": self._records_since_check,
            "check_every": self.check_every,
        }

    @classmethod
    def build_default(
        cls,
        reference_mean: float = 0.5,
        reference_std: float = 0.1,
        z_threshold: float = 3.0,
        buffer_size: int = 500,
        check_every: int = 10,
        actions: Optional[List[AdaptationAction]] = None,
        threshold_updater: Optional[Callable] = None,
        retrain_handler: Optional[Callable] = None,
        feedback_validator: Optional[FeedbackValidator] = None,
        on_record: Optional[Callable[[FeedbackRecord], None]] = None,
        min_adaptation_interval: float = 0.0,
    ) -> "AdaptiveLayer":
        buffer = MemoryBuffer(max_size=buffer_size)
        logger = FeedbackLogger(buffer=buffer, validator=feedback_validator, on_record=on_record)
        detector = DriftDetector(
            strategies=[
                MeanShiftDetector(
                    reference_mean=reference_mean,
                    reference_std=reference_std,
                    z_threshold=z_threshold,
                )
            ]
        )
        engine = AdaptationEngine(
            buffer=buffer,
            threshold_updater=threshold_updater,
            retrain_handler=retrain_handler,
            actions=actions,
            min_interval_seconds=min_adaptation_interval,
            initial_threshold=z_threshold,
        )
        return cls(buffer=buffer, logger=logger, detector=detector, engine=engine, check_every=check_every)

def run_auditing_model(csv_path: str):

    if str(csv_path).lower().endswith(".parquet"):
        data = pd.read_parquet(csv_path)
    else:
        data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    X = data[["Debit", "Credit", "Amount", "Category", "Transaction_Type", "Payment_Method"]].copy()
    data["Balance_Diff"] = data["Balance"] - data["Previous_Balance"]
    X["Balance_Diff"] = data["Balance_Diff"]

    encoder = LabelEncoder()
    for col in ["Category", "Transaction_Type", "Payment_Method"]:
        X[col] = encoder.fit_transform(X[col].astype(str))

    X_scaled = RobustScaler().fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    data["Anomaly_IF"] = iso.fit_predict(X_scaled)

    X_scaled_shuffled = shuffle(X_scaled, random_state=42)

    input_dim = X_scaled_shuffled.shape[1]
    inp = Input(shape=(input_dim,))
    enc = Dense(16, activation="relu")(inp)
    enc = Dense(8,  activation="relu")(enc)
    dec = Dense(16, activation="relu")(enc)
    out = Dense(input_dim, activation="linear")(dec)

    noise = np.random.normal(0, 0.08, X_scaled_shuffled.shape)
    X_noisy = X_scaled_shuffled + noise

    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_noisy, X_scaled_shuffled, epochs=20, batch_size=32,
           validation_split=0.1, verbose=0)

    reconstructions = ae.predict(X_scaled, verbose=0)
    mse = np.mean((X_scaled - reconstructions) ** 2, axis=1)
    threshold = np.percentile(mse, 80)

    data["Anomaly_AE"]    = mse > threshold
    data["Final_Anomaly"] = (data["Anomaly_IF"] == -1) & (data["Anomaly_AE"])

    print(f"\n[AuditingModel] Total anomalies : {data['Final_Anomaly'].sum()}")
    print(f"[AuditingModel] Anomaly rate    : {data['Final_Anomaly'].mean():.2%}")

    return data, mse, threshold

def run_integration(csv_path: str):
    print("=" * 60)
    print("  STEP 1 â€” Running Auditing Model")
    print("=" * 60)
    data, mse, ae_threshold = run_auditing_model(csv_path)
    normal_mse = mse[~data["Final_Anomaly"].values]
    ref_mean   = float(np.mean(normal_mse))
    ref_std    = float(np.std(normal_mse))
    print(f"\n[AdaptiveLayer] Reference  mean={ref_mean:.5f}  std={ref_std:.5f}")

    adaptation_log: List[dict] = []

    def on_drift_action(drift_result: DriftResult):
        adaptation_log.append({
            "score": drift_result.score,
            "reason": drift_result.reason,
            "time": time.strftime("%H:%M:%S"),
        })
        print(f"    Drift action fired | {drift_result.reason}")

    layer = AdaptiveLayer.build_default(
        reference_mean=ref_mean,
        reference_std=ref_std,
        z_threshold=2.5,
        buffer_size=500,
        check_every=50,
        actions=[
            AdaptationAction(name="log_drift", handler=on_drift_action)
        ],
        min_adaptation_interval=0.0,
    )

    print("\n" + "=" * 60)
    print("  STEP 2 â€” Streaming MSE scores through AdaptiveLayer")
    print("=" * 60)

    events: List[AdaptationEvent] = []
    for i, (score, is_anomaly) in enumerate(zip(mse, data["Final_Anomaly"])):
        event = layer.process(
            model_output=float(score),
            feedback=bool(is_anomaly),           # ground truth label
            metadata={"row_index": i, "is_anomaly": bool(is_anomaly)},
        )
        if event is not None:
            events.append(event)
            print(f"  [row {i:>5}] AdaptationEvent â€” actions: {event.actions_taken} | "
                  f"drift score: {event.drift_result.score:.3f}")

    print(f"\n[AdaptiveLayer] Status after streaming:")
    status = layer.status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("  STEP 3 â€” Plotting results")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("AuditingModel أ— AdaptiveLayer â€” Integration Test", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(mse, bins=60, color="steelblue", alpha=0.7, label="All transactions")
    ax.axvline(ae_threshold, color="red",    lw=2, linestyle="--", label=f"AE threshold (p80={ae_threshold:.4f})")
    ax.axvline(ref_mean,     color="orange", lw=2, linestyle="-",  label=f"Normal mean ({ref_mean:.4f})")
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("MSE");  ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    normal_idx  = np.where(~data["Final_Anomaly"].values)[0]
    anomaly_idx = np.where( data["Final_Anomaly"].values)[0]
    ax.scatter(normal_idx,  mse[normal_idx],  s=4,  alpha=0.4, color="steelblue", label="Normal")
    ax.scatter(anomaly_idx, mse[anomaly_idx], s=12, alpha=0.8, color="red",       label="Anomaly")
    ax.axhline(ae_threshold, color="red", lw=1.5, linestyle="--", label="AE threshold")
    for ev in events:
        rows = [r.metadata["row_index"] for r in layer.buffer.all()
                if r.metadata.get("row_index") is not None]
        if rows:
            ax.axvline(max(rows), color="purple", lw=1, alpha=0.6)
    ax.set_title("MSE per Transaction (coloured by Final_Anomaly)")
    ax.set_xlabel("Transaction index");  ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    counts = {
        "Normal":         int((~data["Final_Anomaly"]).sum()),
        "IF only":        int(((data["Anomaly_IF"] == -1) & ~data["Anomaly_AE"]).sum()),
        "AE only":        int((data["Anomaly_AE"]  &  (data["Anomaly_IF"] != -1)).sum()),
        "Both (Final)":   int(data["Final_Anomaly"].sum()),
    }
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["steelblue", "orange", "gold", "red"])
    ax.bar_label(bars, padding=3)
    ax.set_title("Anomaly Detection Breakdown")
    ax.set_ylabel("Transaction count")
    ax = axes[1, 1]
    if events:
        scores = [e.drift_result.score for e in events]
        ax.plot(scores, marker="o", color="purple", lw=1.5)
        ax.axhline(2.5, color="red", linestyle="--", lw=1.5, label="z-threshold=2.5")
        ax.set_title("Drift Score at Each Adaptation Event")
        ax.set_xlabel("Event #");  ax.set_ylabel("Z-score")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No adaptation events fired\n(no significant drift detected)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")
        ax.set_title("Drift Score at Each Adaptation Event")

    plt.tight_layout()
    plt.savefig("adaptive_auditing_results.png", dpi=150)
    plt.show()
    flagged = data[data["Final_Anomaly"]].copy()
    flagged["MSE"] = mse[data["Final_Anomaly"].values]
    print("\n[Summary] First 10 flagged transactions:")
    print(flagged[["MSE", "Anomaly_IF", "Anomaly_AE", "Final_Anomaly"]].head(10).to_string())

    return data, layer, events

if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run the Evolytix auditing integration.")
    parser.add_argument(
        "--web-json",
        action="store_true",
        help="Emit a final JSON payload for the web server.",
    )
    parser.add_argument(
        "--csv",
        default=r"C:\Users\seifh\PythonProject\Evolytix\financial_accounting_cleaned.csv",
        help="CSV file to audit.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where adaptive_auditing_results.png should be written.",
    )
    args = parser.parse_args()

    if args.web_json:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)
        plt.show = lambda *show_args, **show_kwargs: None

        data_out, adaptive_layer, adaptation_events = run_integration(args.csv)
        flagged = data_out[data_out["Final_Anomaly"]].copy()
        payload = {
            "summary": {
                "totalRows": int(len(data_out)),
                "finalAnomalyCount": int(data_out["Final_Anomaly"].sum()),
                "anomalyRate": float(data_out["Final_Anomaly"].mean()),
                "adaptationEvents": int(len(adaptation_events)),
                "bufferSize": int(len(adaptive_layer.buffer)),
            },
            "adaptiveLayer": {
                "status": adaptive_layer.status(),
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "actionsTaken": event.actions_taken,
                        "driftScore": event.drift_result.score,
                        "reason": event.drift_result.reason,
                    }
                    for event in adaptation_events
                ],
            },
            "flaggedRows": flagged.head(10).replace({np.nan: None}).to_dict(orient="records"),
            "plotFile": "adaptive_auditing_results.png",
        }
        print("WEB_JSON:" + json.dumps(payload, default=str))
        sys.exit(0)

    CSV_PATH = r"C:\Users\seifh\PythonProject\Evolytix\financial_accounting_cleaned.csv"

    data_out, adaptive_layer, adaptation_events = run_integration(CSV_PATH)

    print("\nâœ“ Integration test complete.")
    print(f"  Adaptation events fired : {len(adaptation_events)}")
    print(f"  Final anomaly count     : {data_out['Final_Anomaly'].sum()}")
    print(f"  Buffer size             : {len(adaptive_layer.buffer)}")
