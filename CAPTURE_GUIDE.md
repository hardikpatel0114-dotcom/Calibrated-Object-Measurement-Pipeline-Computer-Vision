# Capture Guide — iPhone 12 Pro Max

Follow this exactly. The whole pipeline depends on these photos being consistent.

## 0. Camera settings (do ONCE, then never change)
- **Use the main lens at 1x only.** Never switch to 0.5x (ultra-wide) or 2.5x (tele).
  A different lens = a different calibration and everything breaks.
- **Settings → Camera → Formats → Most Compatible** (saves JPEG, not HEIC — easier for OpenCV).
- **Lock focus + exposure:** tap and hold on the subject until **AE/AF LOCK** appears.
  This stops the phone from silently re-focusing between shots.
- **Flash off.** Use bright, even room light. Avoid harsh shadows and glare.
- **Keep ONE orientation** (hold the phone landscape) for *all* photos — checkerboard
  and cover — so every image is the same pixel size.
- **Transfer at FULL resolution.** Do **not** send via WhatsApp "Photo" — it downscales
  and resizes images inconsistently, which invalidates the calibration (the object photos
  must be the exact same resolution as the calibration photos). Use a **USB cable**
  (File Explorer → *Apple iPhone → Internal Storage → DCIM*), **Google Drive/Photos** at
  *Original quality*, or WhatsApp **"Document"** mode (sends the original file).

---

## Part A — Calibration photos (checkerboard) → target: 20–30 shots
1. On your **laptop**, open `calibration/checkerboard.html`, press **F** for fullscreen.
2. Turn laptop brightness **up**; dim the room to avoid screen glare/reflections.
3. With the iPhone at **1x**, photograph the screen 20–30 times. Vary things:
   - **Angle:** straight-on, then tilted left / right / up / down (about 15–45°).
   - **Distance:** some closer, some farther.
   - **Position in frame:** put the board center, then top-left, bottom-right, edges —
     lens distortion is strongest at the frame edges, so we need coverage there.
   - **Keep the WHOLE board in frame** every time (all squares visible).
   - Keep it **sharp** — no motion blur. If you see rainbow shimmer (moiré), back up a little.
4. (Optional) Measure one square on-screen with a ruler and tell me the mm. Not required.
5. Put these in one folder. We'll rename to `calib_001.jpg`, `calib_002.jpg`, …

## Part B — Object photos (target object) → target: 70+ shots
**Target object:** one flat, rigid object — a **phone cover** is ideal (thin, so its top
surface sits ~level with the reference card = accurate). A slim calculator also works.

**Setup:**
- **Contrasting background:** if the object is dark, use a **light** surface (white paper);
  if light, use your **dark** surface. High contrast = near-perfect auto-labelling.
- Lay the object **flat**. Put your **bank / ID / gift card flat in the frame too**, on the
  same surface beside the object — it is the size reference. **Keep the card in every photo.**
- **Shoot top-down:** phone held flat, directly above, camera parallel to the table
  (not tilted). Keep a fairly consistent height (~30–50 cm).
- Even lighting, no glare on the object or the card.

**Take 70+ photos, varying:**
- Object **position** in the frame (center, corners, edges) and **rotation** (0°, 45°, 90°…).
- **Distance / height** slightly (some closer, some farther).
- **Background** (try a couple of different plain surfaces) and lighting direction.
- Move the card around the frame too.
- Goal: diverse scenes so the model generalizes. Every shot includes the object + the card.

## Part C — Ground-truth measurement (accuracy validation)
- Measure the object's true **outer width (mm)** and **height (mm)** with your calliper.
  Take each measurement 2–3 times and average. This single, precise measurement is the
  **ground truth**. Send me the two numbers.
- We satisfy the "10+ instances" requirement by evaluating the system on **10+ different
  photos** of the object (varied placement/rotation/distance) and comparing each system
  output to this calliper ground truth → reported as MAE and MPE. (This is a deliberate,
  documented interpretation: it measures accuracy *and* repeatability.)
- *(Optional, stronger table)* If you have 1–2 more flat rigid objects of different sizes,
  calliper them too and include a few photos of each — gives the accuracy table size variety.

---

## Handing files to me
Drop the photos into these folders (I'll script the renaming/splitting):
- Calibration → `calibration/images/`
- Objects → `dataset/raw/`

Then tell me "photos are in" and I take over.
