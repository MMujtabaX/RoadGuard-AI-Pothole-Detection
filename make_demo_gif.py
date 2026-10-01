"""Turn a tracked output video into a README-friendly GIF (~7.5 fps, ~10 s, ~5 MB).

Usage:
    python make_demo_gif.py results/tracked_output.mp4 assets/demo.gif
"""
import sys
import cv2
from PIL import Image

src = sys.argv[1] if len(sys.argv) > 1 else "results/tracked_output.mp4"
dst = sys.argv[2] if len(sys.argv) > 2 else "assets/demo.gif"
WIDTH, EVERY_N, MAX_FRAMES, COLORS = 560, 4, 75, 64

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frames, i = [], 0
while len(frames) < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    if i % EVERY_N == 0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (WIDTH, int(h * WIDTH / w)))
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).quantize(colors=COLORS))
    i += 1

frames[0].save(dst, save_all=True, append_images=frames[1:],
               duration=int(1000 * EVERY_N / fps), loop=0, optimize=True)
print(f"Saved {dst} ({len(frames)} frames)")
