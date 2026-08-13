# Frame Awareness Pipeline

static-camera maintaining frame awareness using YOLO26s, OC-SORT, vehicle
motion, and a 15-frame temporal decision. It reports:

- whether a person is present and how many stable person identities exist;
- whether an animal is present and how many stable animal identities exist;
- whether a **moving** vehicle is present and how many stable moving vehicles exist;
- a combined `relevant_present` state.

Cars, motorcycles, buses, and trucks are pooled into `vehicle`. A stationary vehicle
is deliberately excluded from moving-vehicle awareness. The tuned defaults use
YOLO26s at `imgsz=640`, run detection on every processed frame, and target 30 FPS.

## Architecture

```text
camera / RTSP / video
        │
        ▼
runner: timestamps, pacing, reconnect, latest-frame-only live capture
        │
        ▼
FrameAwarenessPipeline.process(frame, timestamp)
        │
        ├─ YOLO26s
        ├─ COCO classes → person / animal / pooled vehicle
        ├─ cross-class vehicle NMS
        ├─ one OC-SORT tracker per awareness group
        ├─ robust vehicle translation + approach/departure motion
        └─ rolling 15-frame presence and counting decision
        │
        ▼
AwarenessResult → callback/JSONL/optional annotated video
```

The frame pipeline never opens a camera. This is an intentional integration boundary:

```python
from frame_awareness import FrameAwarenessPipeline, load_config

config = load_config()
pipeline = FrameAwarenessPipeline(config, processing_fps=30)
pipeline.warmup()

result = pipeline.process(frame, timestamp_seconds)
print(result.person_present, result.moving_vehicle_present)
```

`runner.py` is the standalone wrapper for USB cameras, RTSP streams, and files. Live
capture uses one replaceable latest-frame slot, so slow processing drops stale frames
instead of accumulating delay.

## Repository layout

```text
configs/config.yaml               all tunable runtime settings
src/frame_awareness/config.py     validation and path resolution
src/frame_awareness/types.py      stable typed API
src/frame_awareness/detector.py   YOLO, class pooling, vehicle NMS
src/frame_awareness/tracker.py    OC-SORT and ID management
src/frame_awareness/awareness.py  motion and temporal objective
src/frame_awareness/pipeline.py   reusable single-frame engine
src/frame_awareness/runner.py     camera/file lifecycle and optional output
src/models/yolo26s.pt             default PyTorch model
src/models/yolo26s.onnx           optional ONNX model
```

## Installation

Python 3.10+ and an NVIDIA driver compatible with your PyTorch CUDA build are
recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The model binaries are stored under `src/models` as requested. They are 20 MB and
37 MB, both below GitHub's 100 MB per-file limit. Git LFS can be introduced later if
additional or larger model variants are added.

## Run

USB camera:

```bash
python main.py source.uri=0 source.kind=live runtime.device=0
```

RTSP:

```bash
python main.py \
  source.uri='rtsp://user:password@camera/stream' \
  source.kind=live runtime.device=0
```

Video file:

```bash
python main.py \
  source.uri=/data/example.mp4 source.kind=file runtime.device=0 \
  output.jsonl.enabled=true \
  output.annotated_video.enabled=true
```

Hydra writes the resolved run configuration and runtime outputs under
`outputs/YYYY-MM-DD/HH-MM-SS/`. Use an explicit directory with:

```bash
python main.py source.uri=/data/example.mp4 source.kind=file \
  hydra.run.dir=outputs/my_run
```

## Awareness semantics

At 30 FPS, the default 15-frame window is approximately 0.5 seconds.

| Output | Default rule |
|---|---|
| Person present | person detection ≥0.25 in at least 8 of 15 frames |
| Animal present | animal detection ≥0.10 in at least 8 of 15 frames |
| Person/animal count | stable ID in 8 frames with 4 detector confirmations |
| Moving vehicle | stable moving ID in 6 frames with 3 confirmations ≥0.25 |

Person and animal **presence** intentionally survives OC-SORT ID changes. Their
counts require stable identities. Vehicle presence requires a stable vehicle track
that the motion classifier labels moving.

## Hydra tuning guide

All production controls live in [`configs/config.yaml`](configs/config.yaml). Override
any value from the command line without editing code:

```bash
python main.py detector.image_size=960
python main.py tracker.maximum_age_seconds=0.75
python main.py tracker.association.match_threshold=0.65
python main.py motion.translation.moving_box_diagonals_per_second=0.22
python main.py awareness.person.presence_detection_frames=6
```

Invalid combinations fail before the model or camera starts. Validation covers model
existence, stride alignment, confidence ordering, motion hysteresis, positive max age,
and temporal requirements.

| Parameter | Default | Increasing it | Decreasing it |
|---|---:|---|---|
| `detector.image_size` | 640 | better small-object recall; more GPU latency/memory | faster; more misses and ID fragmentation |
| `confidence.person` | 0.25 | fewer weak people; more missed people | greater recall; more false detections |
| `confidence.vehicle` | 0.25 | fewer weak vehicles | more weak/noisy vehicles |
| `match_threshold` | 0.70 | more permissive in Ultralytics' cost convention | stricter; more ID fragmentation |
| `new_track` | 0.25 | fewer weak new tracks | more tracks and possible duplicates |
| `maximum_age_seconds` | 0.5 | better recovery; more stale identities | quicker removal; more ID changes |
| translation moving threshold | 0.18 | harder to call moving | more sensitivity to jitter |
| scale moving threshold | 0.15 | harder to detect approach/departure | more sensitivity to box-scale noise |
| temporal evidence | 8/15 | steadier but slower activation | faster but more flicker-prone |

`maximum_age_seconds` is OC-SORT's `max_age`. It is converted to tracker frames from
the effective processing FPS. At 30 FPS, 0.5 seconds equals 15 frames.

## Choosing detector image size

`detector.image_size` is YOLO's inference resolution, not the camera resolution. It
must be divisible by the YOLO26s stride of 32. Use `1216`, not `1200`; Ultralytics
would otherwise round 1200 internally.

| Size | Advantage | Tradeoff |
|---:|---|---|
| 640 | lowest tested latency | weaker small-object recall and more fragmented IDs |
| 960 | balanced recall, tracking, and latency | moderate GPU cost |
| 1216 | best tested recall and example-8 person continuity | highest latency and least GPU margin |

Labeled Phase-2 recall at confidence 0.25 and IoU ≥0.50:

| Target | 640 | 960 | 1216 |
|---|---:|---:|---:|
| Person | 77.0% | 81.9% | 87.4% |
| Pooled vehicle | 32.7% | 45.6% | 51.1% |
| Motorcycle | 19.2% | 33.5% | 39.9% |

RTX 2080 SUPER core latency (YOLO + OC-SORT + awareness):

| Video | 640 | 960 | 1216 |
|---|---:|---:|---:|
| Example 6 | 17.52 ms (640-tuned) | 18.92 ms | 23.90 ms |
| Example 7 | 14.84 ms (640-tuned) | 17.36 ms | 23.45 ms |
| Example 8 | 13.14 ms (640-tuned) | 15.41 ms | 21.09 ms |

Example 8 identity continuity before the 640-specific retuning:

| Size | Person IDs | Mean ID lifetime |
|---:|---:|---:|
| 640 | 107 | 54 frames |
| 960 | 72 | 82 frames |
| 1216 | 58 | 99 frames |

The tuned 640 tracker reduced example-8 person IDs to 94 and increased mean lifetime
to 60 frames. It recovered some, but not all, of the resolution loss. These identity
statistics are output-stability measurements, not ground-truth accuracy.

Larger inference input:

- makes small/distant objects occupy more network pixels;
- often improves localization, confidence, and identity continuity;
- consumes more GPU compute and memory approximately with image area;
- may expose more weak detections and false positives;
- may require tracker and motion retuning because box trajectories change;
- cannot recover detail absent from the original camera image;
- does not reduce or increase source decode and video-encoding costs directly.

Start at 640 for the tuned low-latency defaults. Move to 960 if small-object recall or
identity stability is inadequate. Revalidate latency and tracker/motion behavior after
any large resolution change.

## Pixel-size operating guidance

At `imgsz=640`, confidence 0.25, and the strict criterion of ≥90% recall with at
least 100 labeled observations and five IDs:

- person: reliable bin is **64–96 model pixels high**;
- pooled vehicle: no supported bin reached 90%; recall was 88.5% at 48–64 model
  pixels on the shortest box side;
- animal: unknown because no box-labeled animal dataset was available.

Convert model pixels to source pixels with:

```text
gain = min(image_size / source_width, image_size / source_height)
source_pixels = model_pixels / gain
```

For 64 model pixels at `imgsz=640`: approximately 128 source pixels at 1280×720,
192 at 1920×1080, and 384 at 3840×2160. Pixel size is not physical distance;
distance requires camera calibration and scene geometry.

## Vehicle motion

The static-camera classifier uses detector-confirmed observations only. Across a
robust 10-observation history it estimates:

- bottom-center translation in pixels/s, image diagonals/s, and box diagonals/s;
- approach/departure as the robust slope of `log(box height)` per second;
- consistency of scale direction to reject detector jitter.

Default hysteresis:

```text
translation ≤ 0.072 box diagonals/s  → stationary region
translation ≥ 0.18 box diagonals/s   → moving region
scale ≤ 0.06 abs(log-height)/s       → stationary region
scale ≥ 0.15 abs(log-height)/s       → moving region
```

The absolute translation noise floors are 8 pixels/s and 0.0015 image diagonals/s.
Intermediate evidence remains uncertain or preserves the previous stable state.

## Output schema

```json
{
  "schema_version": "1.0",
  "frame_index": 192,
  "timestamp_seconds": 6.4,
  "person_present": true,
  "person_count": 2,
  "animal_present": false,
  "animal_count": 0,
  "moving_vehicle_present": true,
  "moving_vehicle_count": 1,
  "relevant_present": true,
  "tracks": [],
  "latency": {
    "detector_ms": 12.8,
    "tracker_ms": 1.9,
    "awareness_ms": 0.3,
    "total_ms": 15.7
  }
}
```

Disable `output.include_tracks` when only aggregate awareness is needed.

## PyTorch and ONNX

PyTorch is the validated default:

```yaml
detector.backend: pytorch
```

Select the included ONNX file with:

```bash
python main.py detector.backend=onnx
```

ONNX is included for convenient deployment, but backend parity depends on installed
ONNX Runtime providers, CUDA versions, preprocessing, NMS, and precision. Validate
confidence/box parity and latency on the target machine before treating ONNX as an
interchangeable production backend.

## Operational behavior

- Live sources use a bounded latest-frame slot—there is no unbounded frame queue.
- RTSP/camera reads reconnect after repeated failures.
- A long interruption resets tracker and temporal state.
- `SIGINT` and `SIGTERM` request clean shutdown and close capture/output resources.
- Model warm-up occurs before latency collection.
- JSONL and annotated video are disabled by default to protect live latency.
- Every run saves aggregate latency and the resolved Hydra configuration.

For health monitoring, watch processed FPS, p95 latency, stream stalls/reconnects,
and the age of the most recently processed frame.

## Limitations

- The camera must be static. Camera shake, PTZ motion, or electronic stabilization can
  make stationary vehicles appear to move.
- Confidence values are not calibrated probabilities.
- Small objects below the measured pixel ranges have sharply lower recall.
- The animal class has no validated size/accuracy benchmark; one experiment produced
  suspiciously high animal output and requires labeled validation.
- Motorcycle exact-class recall remains weak, although pooled-vehicle awareness helps.
- Grouped vehicle NMS can suppress two genuinely overlapping vehicles.
- OC-SORT has no appearance ReID here and may change IDs after long occlusion or close
  interaction. Counts are therefore less reliable than presence.
- A stationary vehicle is intentionally excluded, regardless of detector confidence.
- The 640 tracker/motion settings were chosen from output stability and agreement with
  prior behavior, not oracle FNR/FPR.
- The 15-frame objective assumes approximately 30 processed FPS. Dropped live frames
  can make those 15 observations span more than 0.5 seconds.
- Saved-video throughput includes decode, drawing, resizing, and software encoding and
  is not representative of the inference core.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

Tests cover configuration invariants, class pooling, grouped NMS, temporal semantics,
FPS-normalized motion, and OC-SORT maximum-age conversion. A deployment release should
also run a short GPU smoke video and an RTSP reconnect test on the target camera/network.
