"""
RoadGuard AI — Pothole Detection & Reporting Dashboard
========================================================
A self-contained Streamlit app that wraps the fine-tuned YOLOv8 pothole detector
(from the project notebooks) behind a simple web UI. Runs entirely on CPU — no
GPU required for inference with the Nano model.

Run with:
    streamlit run streamlit_app.py

Expects the fine-tuned weights at: ./weights/pothole_best.pt
(falls back to a clearly-labeled demo mode using stock COCO weights if missing,
so the app is still explorable before you've trained your own model.)
"""

import os
import io
import time
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "weights", "pothole_best.pt")
FALLBACK_MODEL_PATH = "yolov8n.pt"  # generic COCO weights, used only if fine-tuned model is missing

SEVERITY_COLORS_RGB = {"Low": (46, 160, 67), "Medium": (255, 140, 0), "High": (214, 39, 40)}

st.set_page_config(page_title="RoadGuard AI — Pothole Detection", page_icon="🛣️", layout="wide")


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model...")
def load_model():
    """Returns (model, is_finetuned, error_message). model is None if loading failed."""
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH), True, None
        except Exception as e:
            return None, False, f"Found {MODEL_PATH} but failed to load it: {e}"
    # Fall back to stock COCO weights so the UI is still explorable before you've trained your own model.
    try:
        return YOLO(FALLBACK_MODEL_PATH), False, None
    except Exception as e:
        return None, False, (
            "No fine-tuned model found, and the fallback demo weights couldn't be downloaded "
            f"(needs internet on first run). Error: {e}"
        )


def severity_from_bbox(x1, y1, x2, y2, frame_w, frame_h):
    rel_area = ((x2 - x1) * (y2 - y1)) / float(frame_w * frame_h)
    if rel_area >= 0.05:
        return "High"
    elif rel_area >= 0.015:
        return "Medium"
    return "Low"


def draw_detections(img_bgr, boxes_xyxy, confs, frame_w, frame_h, class_names=None, track_ids=None):
    """Draws colored boxes + severity labels on a BGR image (in place) and returns per-box records."""
    records = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        conf = confs[i]
        severity = severity_from_bbox(x1, y1, x2, y2, frame_w, frame_h)
        color_rgb = SEVERITY_COLORS_RGB[severity]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)
        tid = f"#{track_ids[i]} " if track_ids is not None else ""
        label = f"{tid}{severity} ({conf:.2f})"
        cv2.putText(img_bgr, label, (int(x1), max(int(y1) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

        records.append({
            "track_id": track_ids[i] if track_ids is not None else None,
            "confidence": round(float(conf), 3),
            "severity": severity,
            "bbox_x1": round(float(x1), 1), "bbox_y1": round(float(y1), 1),
            "bbox_x2": round(float(x2), 1), "bbox_y2": round(float(y2), 1),
        })
    return records


def make_demo_gif_bytes(video_path, width=560, every_n=4, max_frames=75, colors=64):
    """Convert an annotated video into a README-friendly GIF (~7.5 fps, ~10 s, ~5 MB)."""
    import io
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (width, int(h * width / w)))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).quantize(colors=colors))
        i += 1
    cap.release()
    if not frames:
        return None
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(1000 * every_n / fps), loop=0, optimize=True)
    return buf.getvalue()


def severity_badge(severity, count):
    color = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[severity]
    return f"{color} **{severity}**: {count}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🛣️ RoadGuard AI")
st.sidebar.caption("AI-powered pothole detection & road damage reporting")

model, is_finetuned, load_error = load_model()

if model is None:
    st.sidebar.error(load_error)
    st.error(
        "**The model couldn't be loaded.** " + load_error +
        "\n\nFix: train the model in `notebooks/01_Train_Pothole_Detector.ipynb` and place the "
        "resulting `pothole_best.pt` at `app/weights/pothole_best.pt`, then restart the app."
    )
    st.stop()

if is_finetuned:
    st.sidebar.success("Fine-tuned pothole model loaded ✅")
else:
    st.sidebar.warning(
        "No fine-tuned model found at `weights/pothole_best.pt` — running with generic "
        "COCO weights as a placeholder. Detections will **not** be real potholes yet. "
        "Train the model in `notebooks/01_Train_Pothole_Detector.ipynb` and drop the "
        "resulting `pothole_best.pt` into `app/weights/`."
    )

conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Severity legend**\n\n"
    "🟢 Low — small / distant damage\n\n"
    "🟠 Medium — noticeable damage\n\n"
    "🔴 High — large / close damage, prioritize repair"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Severity is estimated from how much of the camera frame the pothole's bounding box "
    "covers — a simple, explainable proxy for size/distance, not a calibrated depth measurement."
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Road Infrastructure Damage Monitoring")
st.write(
    "Upload a road photo or a short driving video to detect potholes, estimate repair "
    "priority, and generate a downloadable maintenance report — the kind of tool a city "
    "road department could run on dash-cam footage instead of waiting for citizen complaints."
)

tab_image, tab_video, tab_about = st.tabs(["📷 Image Detection", "🎥 Video Tracking & Report", "ℹ️ About"])

# --------------------------------------------------------------------------
# TAB 1 — Image detection
# --------------------------------------------------------------------------
with tab_image:
    uploaded_img = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"], key="img_uploader")

    if uploaded_img is not None:
        pil_img = Image.open(uploaded_img).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        with st.spinner("Running detection..."):
            t0 = time.time()
            result = model(img_bgr, conf=conf_threshold, verbose=False)[0]
            elapsed = time.time() - t0

        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
        confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))

        annotated = img_bgr.copy()
        records = draw_detections(annotated, boxes_xyxy, confs, w, h)
        df = pd.DataFrame(records)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_img, width='stretch')
        with col2:
            st.subheader("Detected")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')

        st.markdown("### Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Potholes detected", len(df))
        m2.metric("Avg confidence", f"{df['confidence'].mean():.2f}" if len(df) else "—")
        m3.metric("Inference time", f"{elapsed*1000:.0f} ms")
        m4.metric("High severity", int((df["severity"] == "High").sum()) if len(df) else 0)

        if len(df):
            sev_counts = df["severity"].value_counts()
            badges = "  |  ".join(severity_badge(s, int(sev_counts.get(s, 0))) for s in ["High", "Medium", "Low"])
            st.markdown(badges)
            with st.expander("Detection details"):
                st.dataframe(df, width='stretch')

            # Downloads
            success, buf = cv2.imencode(".jpg", annotated)
            st.download_button("⬇️ Download annotated image", data=buf.tobytes(),
                                file_name="pothole_detection.jpg", mime="image/jpeg")
            st.download_button("⬇️ Download detection report (CSV)", data=df.to_csv(index=False),
                                file_name="pothole_report.csv", mime="text/csv")
        else:
            st.info("No potholes detected above the current confidence threshold. "
                     "Try lowering the slider in the sidebar.")
    else:
        st.info("Upload an image to get started, or try the **Video Tracking** tab for footage.")

# --------------------------------------------------------------------------
# TAB 2 — Video tracking
# --------------------------------------------------------------------------
with tab_video:
    uploaded_vid = st.file_uploader("Upload a short driving/walking video (mp4)", type=["mp4", "mov", "avi"],
                                     key="vid_uploader")
    max_seconds = st.slider("Max seconds to process (keeps demo fast on CPU)", 5, 60, 20, 5)

    if uploaded_vid is not None:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_in.write(uploaded_vid.read())
        tmp_in.close()

        cap = cv2.VideoCapture(tmp_in.name)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(fps_in * max_seconds)
        cap.release()

        run_btn = st.button("▶️ Run tracking pipeline")

        if run_btn:
            tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(tmp_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

            progress = st.progress(0, text="Processing video...")
            track_records = {}
            frame_times = []
            frame_idx = 0

            results_stream = model.track(
                source=tmp_in.name, conf=conf_threshold, tracker="bytetrack.yaml",
                persist=True, stream=True, verbose=False,
            )

            for result in results_stream:
                if frame_idx >= max_frames:
                    break
                t0 = time.time()
                frame = result.orig_img.copy()
                boxes = result.boxes

                if boxes.id is not None and len(boxes):
                    ids = boxes.id.int().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    recs = draw_detections(frame, xyxy, confs, w, h, track_ids=ids)
                    severity_rank = {'Low': 0, 'Medium': 1, 'High': 2}

                    for r in recs:
                        tid = r["track_id"]
                        r["first_seen_frame"] = frame_idx
                        if tid not in track_records:
                            track_records[tid] = r
                        else:
                            existing = track_records[tid]
                            # Keep the highest severity ever seen for this pothole
                            if severity_rank[r["severity"]] > severity_rank[existing["severity"]]:
                                track_records[tid]["severity"] = r["severity"]
                            # Keep the highest confidence ever seen
                            if r["confidence"] > existing["confidence"]:
                                track_records[tid]["confidence"] = r["confidence"]

                writer.write(frame)
                frame_times.append(time.time() - t0)
                frame_idx += 1
                progress.progress(min(frame_idx / max_frames, 1.0),
                                   text=f"Processing video... frame {frame_idx}/{max_frames}")

            writer.release()
            progress.empty()

            report_df = pd.DataFrame(track_records.values())
            avg_fps = 1.0 / (np.mean(frame_times) + 1e-9) if frame_times else 0.0

            st.success(f"Done — processed {frame_idx} frames.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Unique potholes", len(report_df))
            m2.metric("Avg FPS (this machine)", f"{avg_fps:.1f}")
            m3.metric("Frames analyzed", frame_idx)
            m4.metric("High severity", int((report_df["severity"] == "High").sum()) if len(report_df) else 0)

            # Re-encode for browser playback (mp4v isn't always browser-friendly; try H264 via ffmpeg if available)
            # Re-encode to H.264 so the browser can play it
            import subprocess
            h264_path = tmp_out_path.replace('.mp4', '_h264.mp4')
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', tmp_out_path,
                    '-vcodec', 'libx264', '-acodec', 'aac',
                    '-movflags', '+faststart', h264_path
                ], capture_output=True, check=True)
                st.video(h264_path)
            except (subprocess.CalledProcessError, FileNotFoundError):
                st.info("Video preview unavailable (ffmpeg not installed). "
                        "Download the annotated video below and open it in VLC or any media player.")

            # Downloads are always offered, whether or not ffmpeg is available
            dl1, dl2 = st.columns(2)
            with open(tmp_out_path, 'rb') as vf:
                dl1.download_button(
                    "⬇️ Download annotated video",
                    data=vf.read(),
                    file_name="tracked_output.mp4",
                    mime="video/mp4",
                )
            with st.spinner("Creating demo GIF..."):
                gif_bytes = make_demo_gif_bytes(tmp_out_path)
            if gif_bytes:
                dl2.download_button(
                    "🎞️ Download demo GIF (for README)",
                    data=gif_bytes,
                    file_name="demo.gif",
                    mime="image/gif",
                )

            if len(report_df):
                sev_counts = report_df["severity"].value_counts()
                badges = "  |  ".join(severity_badge(s, int(sev_counts.get(s, 0))) for s in ["High", "Medium", "Low"])
                st.markdown(badges)
                st.bar_chart(report_df["severity"].value_counts())
                with st.expander("Per-pothole detection log"):
                    st.dataframe(report_df, width='stretch')
                st.download_button("⬇️ Download road damage report (CSV)",
                                    data=report_df.to_csv(index=False),
                                    file_name="road_damage_report.csv", mime="text/csv")
            else:
                st.info("No potholes tracked above the current confidence threshold.")

        os.unlink(tmp_in.name)
    else:
        st.info("Upload a short road video to detect and track potholes across frames.")

# --------------------------------------------------------------------------
# TAB 3 — About
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### What this is

A two-part computer vision pipeline built on **Ultralytics YOLOv8**:

1. **Detection** — a YOLOv8 model fine-tuned (transfer learning) on a 7,000+ image open pothole
   dataset to find potholes in road images.
2. **Tracking & reporting** — multi-object tracking (ByteTrack) on video so the same pothole isn't
   counted multiple times as a vehicle drives past it, plus a simple severity heuristic and an
   exportable CSV report.

### Why this problem

Manual road inspection is slow and reactive — authorities typically only learn about a pothole after
a complaint or an accident. A model like this running on existing dash-cams or patrol-vehicle cameras
could turn routine driving into continuous road-condition monitoring, at near-zero marginal cost per
kilometer surveyed.

### Limitations to be transparent about

- Severity is a *bounding-box-area heuristic*, not a calibrated physical measurement — it doesn't
  account for camera height, lens distortion, or vehicle speed.
- The dataset mixes images from several sources/cameras/lighting conditions; performance will vary on
  truly out-of-distribution conditions (e.g. heavy monsoon flooding obscuring potholes).
- This is a portfolio/demo project, not a validated safety system — it should not be the sole input
  to real repair-prioritization decisions without further validation.

### Tech stack

`Ultralytics YOLOv8` · `OpenCV` · `ByteTrack` · `Streamlit` · `Pandas`
        """
    )
