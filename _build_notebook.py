"""
Generate 0_test.ipynb for the MediaPipe → Inspire RH56 hand-mirroring pipeline.
Run once: python3 _build_notebook.py
"""
import json, textwrap

def code(src):
    lines = textwrap.dedent(src).lstrip("\n").splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}

def md(src):
    lines = textwrap.dedent(src).lstrip("\n").splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""
# MediaPipe → Inspire RH56 Hand-Mirroring Pipeline

**Overview**

| Phase | Description |
|-------|-------------|
| 0 | Setup — installs and imports |
| 1 | Feature extraction — joint angles from MediaPipe |
| 2 | Inspire hand interface — Modbus RTU wrapper |
| 3 | Calibration data collection — webcam-guided key poses |
| 4 | Neural network — train & validate MLP |
| 5 | Real-time teleoperation loop |
| 6 | Persistence — save / reload pipeline |

DOF order for all robot position arrays: `[little, ring, middle, index, thumb_bend, thumb_rotation]`
"""))

# ── Phase 0 ────────────────────────────────────────────────────────────────────
cells.append(md("## Phase 0 — Setup"))

cells.append(code("""
%pip install -q mediapipe opencv-python torch numpy matplotlib pymodbus
# Uncomment to install inspirepy from source if preferred over raw pymodbus:
# %pip install -q git+https://github.com/kefhuang/inspirepy.git
"""))

cells.append(code("""
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ── DOF constants ──────────────────────────────────────────────────────────────
# DOF ordering matches the Inspire RH56 register layout
DOF_NAMES = ["Little", "Ring", "Middle", "Index", "Thumb Bend", "Thumb Rot"]
N_DOFS = 6

# All positions are on a 0–1000 integer scale (0 = closed, 1000 = open)
POS_MIN, POS_MAX = 0, 1000

# ── Serial / Modbus defaults (RH56dfq / RH56DFX over USB-CH340) ───────────────
DEFAULT_PORT  = "/dev/ttyUSB0"
DEFAULT_BAUD  = 115200
DEVICE_ID     = 1           # Modbus slave ID (factory default)

# ── Modbus register addresses (per Inspire RH56 manual) ───────────────────────
REG_ANGLE_SET = 1486    # write target angle — 6 consecutive registers
REG_ANGLE_ACT = 1546    # read actual angle  — 6 consecutive registers
REG_FORCE_SET = 1498    # write force limit
REG_SPEED_SET = 1522    # write speed

# ── File paths ─────────────────────────────────────────────────────────────────
CALIB_FILE = Path("calibration_data.npz")
MODEL_FILE  = Path("hand_mapping_model.pt")
"""))

# ── Phase 1 ────────────────────────────────────────────────────────────────────
cells.append(md("""
## Phase 1 — Feature Extraction

Compute **joint angles** (in degrees) from MediaPipe 21-point world landmarks.
Using angles (rather than raw landmark coordinates) makes features invariant to
hand size and camera distance.

MediaPipe landmark index reference:
```
Wrist: 0
Thumb:  CMC=1  MCP=2  IP=3   TIP=4
Index:  MCP=5  PIP=6  DIP=7  TIP=8
Middle: MCP=9  PIP=10 DIP=11 TIP=12
Ring:   MCP=13 PIP=14 DIP=15 TIP=16
Little: MCP=17 PIP=18 DIP=19 TIP=20
```
"""))

cells.append(code("""
# Each triplet (a, b, c) defines the angle measured at vertex b.
JOINT_TRIPLETS = [
    # Thumb — 3 joints
    (0,  1,  2),   # thumb CMC
    (1,  2,  3),   # thumb MCP
    (2,  3,  4),   # thumb IP
    # Index — 3 joints
    (0,  5,  6),   # index MCP
    (5,  6,  7),   # index PIP
    (6,  7,  8),   # index DIP
    # Middle — 3 joints
    (0,  9, 10),   # middle MCP
    (9, 10, 11),   # middle PIP
    (10, 11, 12),  # middle DIP
    # Ring — 3 joints
    (0, 13, 14),   # ring MCP
    (13, 14, 15),  # ring PIP
    (14, 15, 16),  # ring DIP
    # Little — 3 joints
    (0, 17, 18),   # little MCP
    (17, 18, 19),  # little PIP
    (18, 19, 20),  # little DIP
    # Thumb abduction — angle at thumb CMC between thumb MCP and index MCP
    (2,  1,  5),
]
N_FEATURES = len(JOINT_TRIPLETS)   # 16
print(f"Feature vector size: {N_FEATURES}")


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    \"\"\"Angle at vertex b formed by rays b→a and b→c, in degrees [0, 180].\"\"\"
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def compute_joint_angles(landmarks) -> np.ndarray:
    \"\"\"
    Compute the 16-element joint-angle feature vector from MediaPipe hand landmarks.

    Parameters
    ----------
    landmarks : mediapipe NormalizedLandmarkList  OR  array-like of shape (21, 3)

    Returns
    -------
    np.ndarray of shape (N_FEATURES,), angles in degrees
    \"\"\"
    if hasattr(landmarks, "landmark"):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    else:
        pts = np.asarray(landmarks, dtype=float)

    return np.array([
        _angle_deg(pts[a], pts[b], pts[c])
        for a, b, c in JOINT_TRIPLETS
    ])
"""))

cells.append(code("""
# ── Unit test with synthetic hand poses ───────────────────────────────────────

def _synthetic_open_hand():
    \"\"\"21 landmarks for a flat open hand (fingers extended along +y).\"\"\"
    pts = np.zeros((21, 3))
    # wrist at origin; finger MCPs spread along x; tips extend along +y
    starts   = [1, 5, 9, 13, 17]          # thumb, index, middle, ring, little
    offsets_x = [0.10, 0.06, 0.02, -0.02, -0.06]
    for sx, start in zip(offsets_x, starts):
        for j in range(4):
            pts[start + j] = [sx, 0.05 * (j + 1), 0.0]
    return pts


def _synthetic_fist():
    \"\"\"21 landmarks for a clenched fist (joints curled toward palm).\"\"\"
    pts = np.zeros((21, 3))
    starts    = [1, 5, 9, 13, 17]
    offsets_x = [0.10, 0.06, 0.02, -0.02, -0.06]
    for sx, start in zip(offsets_x, starts):
        pts[start]     = [sx,  0.05,  0.00]   # MCP/CMC: forward
        pts[start + 1] = [sx,  0.08, -0.02]   # PIP/MCP: begin curl
        pts[start + 2] = [sx,  0.07, -0.05]   # DIP/IP:  curled back
        pts[start + 3] = [sx,  0.04, -0.06]   # TIP:     tucked under
    return pts


open_angles = compute_joint_angles(_synthetic_open_hand())
fist_angles  = compute_joint_angles(_synthetic_fist())

print(f"Feature vector length : {len(open_angles)}  (expected {N_FEATURES})")
print(f"\\nOpen hand angles (°)  : {np.round(open_angles, 1)}")
print(f"Fist angles       (°)  : {np.round(fist_angles,  1)}")

assert len(open_angles) == N_FEATURES
assert open_angles.mean() > fist_angles.mean(), "Open angles should be larger on average than fist"
print("\\n✓ Feature extraction sanity check passed")
"""))

# ── Phase 2 ────────────────────────────────────────────────────────────────────
cells.append(md("""
## Phase 2 — Inspire Hand Interface

Thin Modbus RTU wrapper around the Inspire RH56.  Falls back to **mock mode**
automatically when no hardware is detected, so all downstream cells work without
a physical hand attached.
"""))

cells.append(code("""
from pymodbus.client import ModbusSerialClient


class InspireHand:
    \"\"\"
    Modbus RTU interface for the Inspire RH56DFX / RH56dfq.

    All position values are integers in [0, 1000].
    DOF order: [little, ring, middle, index, thumb_bend, thumb_rotation]
    \"\"\"

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD,
                 device_id: int = DEVICE_ID):
        self.port      = port
        self.baud      = baud
        self.device_id = device_id
        self._client   = None
        self._mock     = False

    # ── connection ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        \"\"\"Open serial connection. Returns True on success; enables mock mode on failure.\"\"\"
        self._client = ModbusSerialClient(
            port=self.port,
            baudrate=self.baud,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=1,
        )
        ok = self._client.connect()
        if not ok:
            print(f"[InspireHand] WARNING: could not connect on {self.port}. "
                  "Running in mock mode — commands will be silently ignored.")
            self._mock = True
        else:
            print(f"[InspireHand] Connected on {self.port} @ {self.baud} baud.")
        return ok

    def disconnect(self):
        if self._client and not self._mock:
            self._client.close()
        print("[InspireHand] Disconnected.")

    # ── control ────────────────────────────────────────────────────────────────

    def set_angles(self, positions) -> bool:
        \"\"\"
        Write target positions to all 6 DOFs atomically.

        Parameters
        ----------
        positions : sequence of 6 numbers in [0, 1000]
        \"\"\"
        if len(positions) != N_DOFS:
            raise ValueError(f"Expected {N_DOFS} positions, got {len(positions)}")
        clamped = [int(np.clip(p, POS_MIN, POS_MAX)) for p in positions]
        if self._mock:
            return True
        result = self._client.write_registers(
            address=REG_ANGLE_SET, values=clamped, slave=self.device_id
        )
        return not result.isError()

    def get_angles(self) -> list:
        \"\"\"Read actual positions of all 6 DOFs. Returns list of ints in [0, 1000].\"\"\"
        if self._mock:
            return [500] * N_DOFS
        result = self._client.read_holding_registers(
            address=REG_ANGLE_ACT, count=N_DOFS, slave=self.device_id
        )
        if result.isError():
            return [0] * N_DOFS
        return list(result.registers)

    # ── context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
"""))

cells.append(code("""
# ── Quick connection test ─────────────────────────────────────────────────────
# Safe to run without hardware — falls back to mock mode automatically.

hand_test = InspireHand(port=DEFAULT_PORT)
hand_test.connect()

if not hand_test._mock:
    print("Current angles:", hand_test.get_angles())
    hand_test.set_angles([500] * N_DOFS)   # center all DOFs
    time.sleep(0.5)
    print("After set to 500:", hand_test.get_angles())
    hand_test.disconnect()
else:
    print("Mock mode active — hardware tests skipped.")
"""))

# ── Phase 3 ────────────────────────────────────────────────────────────────────
cells.append(md("""
## Phase 3 — Calibration Data Collection

The user is guided through **5 key poses** in front of the webcam.  For each pose,
60 frames of joint angles are recorded alongside the analytically-determined robot
target positions.  The data is saved to `calibration_data.npz`.

| Pose | Robot targets `[little, ring, middle, index, thumb_bend, thumb_rot]` |
|------|-----------------------------------------------------------------------|
| Open Hand  | `[1000, 1000, 1000, 1000, 1000,  500]` |
| Fist       | `[   0,    0,    0,    0,  148,  894]` |
| Point      | `[   0,    0,    0, 1000,  200,  894]` |
| Thumbs Up  | `[   0,    0,    0,    0, 1000,  500]` |
| Pinch      | `[1000, 1000, 1000,    0,    0,  500]` |
"""))

cells.append(code("""
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Calibration poses: (display_name, robot_target_positions)
CALIB_POSES = [
    ("Open Hand",  [1000, 1000, 1000, 1000, 1000,  500]),
    ("Fist",       [   0,    0,    0,    0,  148,  894]),
    ("Point",      [   0,    0,    0, 1000,  200,  894]),
    ("Thumbs Up",  [   0,    0,    0,    0, 1000,  500]),
    ("Pinch",      [1000, 1000, 1000,    0,    0,  500]),
]


class CalibrationCollector:
    \"\"\"
    Displays a webcam feed with on-screen guidance, collects N frames of joint
    angles per pose, and returns paired (features, labels) arrays.
    \"\"\"

    def __init__(self, poses=None, n_frames: int = 60,
                 camera_index: int = 0, min_confidence: float = 0.85):
        self.poses          = poses or CALIB_POSES
        self.n_frames       = n_frames
        self.camera_index   = camera_index
        self.min_confidence = min_confidence

    def collect(self) -> tuple:
        \"\"\"
        Run the interactive calibration session.

        Returns
        -------
        features : np.ndarray, shape (N_poses * n_frames, N_FEATURES)
        labels   : np.ndarray, shape (N_poses * n_frames, N_DOFS)
        \"\"\"
        features_list, labels_list = [], []
        cap = cv2.VideoCapture(self.camera_index)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
            model_complexity=1,
        ) as hands:
            for pose_name, target in self.poses:
                collected = []
                print(f"\\n>>> Pose: '{pose_name}'  —  target positions: {target}")
                print(f"    Hold the pose and keep it steady.")
                print(f"    Collecting {self.n_frames} frames...")

                while len(collected) < self.n_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    n = len(collected)
                    if result.multi_hand_world_landmarks:
                        wlm    = result.multi_hand_world_landmarks[0]
                        angles = compute_joint_angles(wlm)
                        collected.append(angles)

                        mp_draw.draw_landmarks(
                            frame,
                            result.multi_hand_landmarks[0],
                            mp_hands.HAND_CONNECTIONS,
                        )
                        label_text = f"{pose_name}  [{n}/{self.n_frames}]"
                        cv2.putText(frame, label_text, (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 0), 2)
                    else:
                        cv2.putText(frame, f"{pose_name}  — no hand detected",
                                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                    (0, 0, 255), 2)

                    cv2.imshow("Calibration", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        raise RuntimeError("Calibration aborted by user.")

                features_list.extend(collected)
                labels_list.extend([target] * len(collected))
                print(f"    ✓ {len(collected)} frames collected for '{pose_name}'")

        cap.release()
        cv2.destroyAllWindows()

        features = np.array(features_list, dtype=np.float32)
        labels   = np.array(labels_list,   dtype=np.float32)
        return features, labels
"""))

cells.append(code("""
# ── Run calibration ───────────────────────────────────────────────────────────
# A webcam window will open.  Hold each pose steadily until the counter fills.
# Press 'q' in the webcam window at any time to abort.

collector = CalibrationCollector(n_frames=60)
features, labels = collector.collect()

assert features.shape == (len(CALIB_POSES) * 60, N_FEATURES), f"Unexpected features shape: {features.shape}"
assert labels.shape   == (len(CALIB_POSES) * 60, N_DOFS),     f"Unexpected labels shape: {labels.shape}"

np.savez(CALIB_FILE, features=features, labels=labels)
print(f"\\nSaved: features {features.shape}, labels {labels.shape}  →  {CALIB_FILE}")
"""))

# ── Phase 4 ────────────────────────────────────────────────────────────────────
cells.append(md("""
## Phase 4 — Neural Network

A small MLP maps the 16-element joint-angle feature vector to 6 robot DOF
positions.  The `Sigmoid` output is scaled by 1000 so predictions land in
`[0, 1000]` without any explicit clipping.

```
Linear(16 → 32) → ReLU → Linear(32 → 32) → ReLU → Linear(32 → 6) → Sigmoid × 1000
```
"""))

cells.append(code("""
class HandMappingMLP(nn.Module):
    \"\"\"
    Lightweight MLP: joint angles (N_FEATURES) → robot DOF positions (N_DOFS).
    Output is in [0, 1000] via Sigmoid × POS_MAX.
    \"\"\"

    def __init__(self, n_input: int = N_FEATURES, n_output: int = N_DOFS,
                 hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_output),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be pre-normalised to [0, 1] (divide raw angles by 180)
        return self.net(x) * POS_MAX

    @torch.no_grad()
    def predict(self, angles_norm: np.ndarray) -> np.ndarray:
        \"\"\"Numpy convenience wrapper.  Input must already be normalised to [0, 1].\"\"\"
        x = torch.from_numpy(np.atleast_2d(angles_norm).astype(np.float32))
        return self.forward(x).squeeze(0).numpy()
"""))

cells.append(code("""
def train_mapping(features: np.ndarray, labels: np.ndarray,
                  epochs: int = 300, lr: float = 1e-3,
                  plot: bool = True) -> HandMappingMLP:
    \"\"\"Train the MLP on calibration pairs and return the fitted model.\"\"\"
    # Normalise: angles 0–180° → 0–1;  positions 0–1000 → 0–1
    X = torch.from_numpy((features / 180.0).astype(np.float32))
    Y = torch.from_numpy((labels   / POS_MAX).astype(np.float32))

    model     = HandMappingMLP()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        pred = model.net(X)
        loss = criterion(pred, Y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>4d}/{epochs}  |  Loss: {loss.item():.6f}")

    if plot:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(losses)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title("Training Loss"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\\nModel saved → {MODEL_FILE}")
    model.eval()
    return model


# ── Load calibration data and train ───────────────────────────────────────────
data     = np.load(CALIB_FILE)
features = data["features"]
labels   = data["labels"]

model = train_mapping(features, labels, epochs=300)
"""))

cells.append(code("""
def validate_model(model: HandMappingMLP, features: np.ndarray,
                   labels: np.ndarray, poses=None):
    \"\"\"Print per-pose MAE and plot residuals for all DOFs.\"\"\"
    poses = poses or CALIB_POSES
    model.eval()
    n_per_pose = features.shape[0] // len(poses)

    fig, axes = plt.subplots(len(poses), 1,
                              figsize=(9, 2.2 * len(poses)), sharex=True)
    fig.suptitle("Per-Pose Prediction Residuals (predicted − target)", fontsize=12)

    for i, (pose_name, target) in enumerate(poses):
        chunk      = features[i * n_per_pose:(i + 1) * n_per_pose] / 180.0
        preds      = np.array([model.predict(chunk[j:j+1]).squeeze()
                                for j in range(len(chunk))])
        mean_pred  = preds.mean(axis=0)
        residuals  = mean_pred - np.array(target, dtype=float)
        mae        = np.abs(residuals).mean()

        colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in residuals]
        axes[i].bar(DOF_NAMES, residuals, color=colors)
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_ylim(-200, 200)
        axes[i].set_ylabel(pose_name, fontsize=8)
        axes[i].grid(True, axis="y", alpha=0.3)
        print(f"  {pose_name:<12}  MAE = {mae:6.1f}   residuals = {np.round(residuals, 1)}")

    axes[-1].set_xlabel("DOF")
    plt.tight_layout()
    plt.show()


validate_model(model, features, labels)
"""))

# ── Phase 5 ────────────────────────────────────────────────────────────────────
cells.append(md("""
## Phase 5 — Real-Time Teleoperation

Run without hardware first to confirm the DOF bar chart responds correctly to
hand movement before connecting the Inspire hand.

```python
# Display-only (no robot):
run_teleoperation(model)

# With hardware:
with InspireHand(port="/dev/ttyUSB0") as hand:
    run_teleoperation(model, hand=hand)
```
"""))

cells.append(code("""
class EMAFilter:
    \"\"\"Per-channel exponential moving average smoother for DOF positions.\"\"\"

    def __init__(self, n_channels: int, alpha: float = 0.25):
        self.alpha    = alpha
        self.n_channels = n_channels
        self._state   = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(float).copy()
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self):
        self._state = None
"""))

cells.append(code("""
def run_teleoperation(
    model: HandMappingMLP,
    hand: "InspireHand | None" = None,
    camera_index: int = 0,
    target_hz: float = 20.0,
    dead_zone: int = 10,
    ema_alpha: float = 0.25,
):
    \"\"\"
    Real-time hand-mirroring loop.

    Parameters
    ----------
    model        : trained HandMappingMLP (eval mode)
    hand         : InspireHand instance, or None for display-only mode
    camera_index : webcam device index
    target_hz    : maximum command rate sent to the robot
    dead_zone    : skip robot update when max positional change < this value
    ema_alpha    : EMA smoothing coefficient (lower = smoother but laggier)
    \"\"\"
    model.eval()
    ema            = EMAFilter(n_channels=N_DOFS, alpha=ema_alpha)
    prev_positions = np.full(N_DOFS, 500.0)
    dt             = 1.0 / target_hz

    cap = cv2.VideoCapture(camera_index)

    # Live bar chart for DOF positions
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.bar(DOF_NAMES, [500.0] * N_DOFS, color="#3498db")
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Position (0–1000)")
    ax.set_title("Robot DOF Positions")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    chart_update_time = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        model_complexity=1,
    ) as hands:
        print("Teleoperation running — press 'q' in the webcam window to stop.")
        frame_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_world_landmarks:
                wlm       = result.multi_hand_world_landmarks[0]
                angles    = compute_joint_angles(wlm) / 180.0   # normalise
                raw_pos   = model.predict(angles[np.newaxis, :]).squeeze()
                positions = np.clip(ema.update(raw_pos), POS_MIN, POS_MAX)

                # Dead zone — skip command if overall change is negligible
                delta = np.max(np.abs(positions - prev_positions))
                if delta >= dead_zone:
                    if hand is not None:
                        # TODO: thumb collision avoidance
                        # Before sending, constrain thumb_bend (DOF 4) and
                        # thumb_rotation (DOF 5) based on adjacent finger
                        # positions to avoid mechanical collisions.
                        # e.g., when index (DOF 3) is very closed (< 100),
                        # limit thumb rotation to a safe range.
                        hand.set_angles(positions.tolist())
                    prev_positions = positions.copy()

                # Draw landmarks on frame
                mp_draw.draw_landmarks(
                    frame,
                    result.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )
                for i, (name, val) in enumerate(zip(DOF_NAMES, positions)):
                    cv2.putText(frame, f"{name}: {int(val):>4}",
                                (10, 30 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 1)
            else:
                cv2.putText(frame, "No hand detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                positions = prev_positions  # hold last position

            # Update bar chart at ~15 Hz
            now = time.time()
            if now - chart_update_time >= 1.0 / 15:
                for bar, val in zip(bars, positions):
                    bar.set_height(val)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                chart_update_time = now

            cv2.imshow("Teleoperation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Rate-limit
            elapsed = time.time() - frame_time
            wait    = dt - elapsed
            if wait > 0:
                time.sleep(wait)
            frame_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)
    print("Teleoperation stopped.")


# ── Run (display-only; no hardware needed) ─────────────────────────────────────
# run_teleoperation(model)

# ── Run with hardware ──────────────────────────────────────────────────────────
# with InspireHand(port="/dev/ttyUSB0") as hand:
#     run_teleoperation(model, hand=hand)
"""))

# ── Phase 6 ────────────────────────────────────────────────────────────────────
cells.append(md("## Phase 6 — Persistence"))

cells.append(code("""
def load_pipeline(model_path=MODEL_FILE, calib_path=CALIB_FILE):
    \"\"\"
    Reload a previously trained model and calibration dataset.

    Returns
    -------
    model    : HandMappingMLP in eval mode
    features : np.ndarray of shape (N, N_FEATURES)
    labels   : np.ndarray of shape (N, N_DOFS)
    \"\"\"
    m = HandMappingMLP()
    m.load_state_dict(torch.load(model_path, weights_only=True))
    m.eval()
    print(f"Model loaded from        {model_path}")

    data = np.load(calib_path)
    feats, lbls = data["features"], data["labels"]
    print(f"Calibration data loaded  features {feats.shape},  labels {lbls.shape}")
    return m, feats, lbls


# ── Quick-start: skip calibration if a saved pipeline exists ──────────────────
if MODEL_FILE.exists() and CALIB_FILE.exists():
    print("Existing pipeline found.  Loading...")
    model, features, labels = load_pipeline()
    print("\\nReady for teleoperation.  Run:")
    print("  run_teleoperation(model)")
    print("  # — or with hardware —")
    print("  # with InspireHand() as hand: run_teleoperation(model, hand=hand)")
else:
    print("No saved pipeline found.  Run Phases 3–5 to calibrate and train.")
"""))


# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = "/home/katelyn/research-inspire_hand_mirroring/0_test.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {len(cells)} cells to {out_path}")
