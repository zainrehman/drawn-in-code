"""Harmonograph Sigil - four simulated pendulums drive a pen. The same four
pendulums are also the soundtrack (each one is a tone). When they stop... the pen doesn't.
Everything generated here. Output: harmonograph_sigil.mp4 (1080x1920, 30 fps)
"""
import numpy as np, subprocess, os
from scipy import signal
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1313)
SR, FPS, W, H = 44100, 30, 1080, 1920
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
DUR = 34.5
T_DRAW0, T_DRAW1 = 3.0, 25.5          # pendulums swing
T_PEN0, T_PEN1 = 27.0, 29.7           # pen moves on its own
T_OPEN = 30.0
T_END = 31.6

# ------------------------------------------------------------------ pendulums
PEND = [  # amplitude, frequency, phase, damping   (x uses 0,1 ; y uses 2,3)
    (230, 2.000, 0.0, 0.0042), (200, 3.003, 0.0, 0.0052),
    (230, 2.000, np.pi / 2, 0.0042), (200, 3.000, -np.pi / 2, 0.0052)]
T_SIM = 430.0
CX, CY = W / 2, 980
def sim_time(tsec):
    return np.clip((tsec - T_DRAW0) / (T_DRAW1 - T_DRAW0), 0, 1) * T_SIM
def pen_xy(tau):
    p = [a * np.sin(f * tau + ph) * np.exp(-d * tau) for a, f, ph, d in PEND]
    return CX + p[0] + p[1], CY + p[2] + p[3]
def amp(i, tau):
    a, f, ph, d = PEND[i]; return np.exp(-d * tau)

# eye the pen draws alone
def eye_path(u):
    """u in [0,1] -> point. upper lid, lower lid, iris circle."""
    ew, eh, ir = 150, 70, 46
    if u < .35:
        s = u / .35; x = -ew + 2 * ew * s; y = -eh * np.sin(np.pi * s); return CX + x, CY + y
    if u < .7:
        s = (u - .35) / .35; x = ew - 2 * ew * s; y = eh * np.sin(np.pi * s); return CX + x, CY + y
    s = (u - .7) / .3; a = -np.pi / 2 + 2 * np.pi * s * 1.05
    return CX + ir * np.cos(a), CY + ir * np.sin(a)

# ------------------------------------------------------------------ audio
N = int(DUR * SR); t = np.arange(N) / SR
tau_t = sim_time(t)
run = np.interp(t, [0, T_DRAW0 - .4, T_DRAW0, T_DRAW1 - .3, T_DRAW1 + .15, DUR], [0, 0, 1, 1, 0, 0])
mix = np.zeros(N)
BASE = 55.0
for i, (a, f, ph, d) in enumerate(PEND):
    fr = BASE * f * (1 if i % 2 == 0 else 2)
    tone = (np.sin(2 * np.pi * fr * t + ph) + .35 * np.sin(2 * np.pi * 2 * fr * t) + .15 * np.sin(2 * np.pi * 3 * fr * t + 1))
    mix += tone * np.exp(-d * tau_t) * run * .11
# pen scratch follows pen speed
xs, ys = pen_xy(tau_t)
speed = np.hypot(np.gradient(xs), np.gradient(ys)) * SR / 1000
speed = signal.lfilter([.001], [1, -.999], speed)
scr = signal.lfilter(*signal.butter(2, [2000 / (SR / 2), 7000 / (SR / 2)], "band"), rng.standard_normal(N))
scr /= np.max(np.abs(scr))
mix += scr * np.clip(speed / np.percentile(speed[run > .5], 95), 0, 1.2) * run * .06
mix += np.sin(2 * np.pi * 27.5 * t) * run * .08                        # floor hum
# solo pen: dry, close scratching
pen_on = ((t >= T_PEN0) & (t < T_PEN1)).astype(float)
pen_on = signal.lfilter([.01], [1, -.99], pen_on)
grain = (rng.random(N) < .02) * rng.standard_normal(N)
grain = signal.lfilter(*signal.butter(2, [1200 / (SR / 2), 6000 / (SR / 2)], "band"), grain)
mix += (scr * .5 + grain * 3) * pen_on * .22
# the eye opens
i0 = int(T_OPEN * SR); n = int(3.5 * SR); tt = np.arange(n) / SR
boom = np.sin(2 * np.pi * (44 - 8 * tt) * tt) * np.exp(-tt * 1.4) * .8
boom += signal.lfilter(*signal.butter(2, 300 / (SR / 2), "low"), rng.standard_normal(n)) * np.exp(-tt * 3) * 2
cluster = sum(np.sin(2 * np.pi * f * tt) for f in (110, 116.5, 164.8, 174.6, 233.1)) * np.interp(tt, [0, .3, 1.4, 1.6, 3.5], [0, .05, .07, 0, 0])
mix[i0:i0 + n] += boom + cluster
mix = np.tanh(mix * 1.2) * .9
wavfile.write(os.path.join(OUT, "sigil.wav"), SR, (mix * 32767).astype(np.int16))

# ------------------------------------------------------------------ visuals
f_cap = ImageFont.truetype(MONOB, 46)
f_small = ImageFont.truetype(MONO, 26)
f_end = ImageFont.truetype(MONO, 32)
stops = [(0, (6, 3, 3)), (.18, (70, 12, 8)), (.42, (190, 70, 20)), (.7, (245, 170, 70)), (.9, (255, 232, 170)), (1, (255, 252, 240))]
lut = np.zeros((256, 3), np.float32)
for c in range(3):
    lut[:, c] = np.interp(np.linspace(0, 1, 256), [s[0] for s in stops], [s[1][c] for s in stops])
red_lut = lut.copy(); red_lut[:, 1] *= .35; red_lut[:, 2] *= .3

YY, XX = np.mgrid[0:H, 0:W]
vig = (np.clip(1 - (((XX - W / 2) / (W * .78)) ** 2 + ((YY - CY) / (H * .62)) ** 2), 0, 1) ** .55)
paper = Image.fromarray((rng.random((H // 4, W // 4)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))
bg = (np.stack([np.asarray(paper)] * 3, -1).astype(np.float32) / 255 * np.array([10, 6, 4]) + np.array([7, 4, 3])) * vig[..., None]

# invented glyphs for the ring (procedural, not from any real script)
def make_glyph(seed):
    r = np.random.default_rng(seed); pts = [(x, y) for x in (-1, 0, 1) for y in (-1.4, 0, 1.4)]
    strokes = []
    for _ in range(r.integers(2, 5)):
        a, b = r.choice(len(pts), 2, replace=False); strokes.append((pts[a], pts[b]))
    circ = r.random() < .35
    return strokes, circ
GLYPHS = [make_glyph(s) for s in range(24)]
RING_R = 455
def draw_ring(d, tsec, k, red):
    rot = tsec * .05 if tsec < T_OPEN else T_OPEN * .05
    col = (255, 70, 50) if red else (255, 196, 110)
    col = tuple(int(c * k) for c in col)
    for i, (strokes, circ) in enumerate(GLYPHS):
        a = rot + i * 2 * np.pi / len(GLYPHS)
        gx, gy = CX + RING_R * np.cos(a), CY + RING_R * np.sin(a)
        ca, sa = np.cos(a + np.pi / 2), np.sin(a + np.pi / 2)
        s = 11
        for (x1, y1), (x2, y2) in strokes:
            p1 = (gx + (x1 * ca - y1 * sa) * s, gy + (x1 * sa + y1 * ca) * s)
            p2 = (gx + (x2 * ca - y2 * sa) * s, gy + (x2 * sa + y2 * ca) * s)
            d.line([p1, p2], fill=col, width=3)
        if circ:
            d.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], outline=col, width=2)
    for rr in (RING_R - 34, RING_R + 34):
        d.ellipse([CX - rr, CY - rr, CX + rr, CY + rr], outline=tuple(int(c * .55) for c in col), width=2)

CAPS = [(.3, 2.9, "Four pendulums.\nOne pen."),
        (3.2, 7.8, "No hand touches it."),
        (8.6, 14.4, "What you hear is\nthe pendulums."),
        (15.0, 20.4, "Every time I run it,\nit draws this sigil."),
        (20.8, 25.4, "No matter where\nI start them."),
        (25.7, 27.8, "Then they stopped."),
        (27.9, 30.0, "The pen didn't."),
        (30.2, T_END, "")]

acc = np.zeros((H, W), np.float32)
SUB = 60
last_tau = 0.0
pen_last = None
def render(fi):
    global last_tau, pen_last
    tsec = fi / FPS
    # --- advance the pen and add ink
    layer = Image.new("L", (W, H), 0); ld = ImageDraw.Draw(layer)
    tau = sim_time(tsec)
    pen = None
    if tau > last_tau:
        taus = np.linspace(last_tau, tau, SUB)
        px, py = pen_xy(taus)
        ld.line(list(zip(px, py)), fill=34, width=2)
        last_tau = tau; pen = (px[-1], py[-1])
    elif T_DRAW0 <= tsec < T_PEN0:
        pen = pen_xy(T_SIM)
    if T_PEN0 <= tsec < T_PEN1 + 1 / FPS:
        u1 = np.clip((tsec - T_PEN0) / (T_PEN1 - T_PEN0), 0, 1)
        u0 = np.clip((tsec - 1 / FPS - T_PEN0) / (T_PEN1 - T_PEN0), 0, 1)
        us = np.linspace(u0, u1, 20)
        pts = [eye_path(u) for u in us]
        # jittery, hand-like line
        pts = [(x + rng.normal(0, .6), y + rng.normal(0, .6)) for x, y in pts]
        ld.line(pts, fill=255, width=5)
        pen = pts[-1]
    elif T_PEN1 <= tsec < T_OPEN:
        pen = eye_path(1.0)
    acc_add = np.asarray(layer).astype(np.float32) / 255
    acc[:] = acc + acc_add
    # --- tone map + glow
    v = 1 - np.exp(-acc * 1.6)
    if tsec >= T_DRAW1:             # sigil cools while the pendulums rest
        v = v * (1 - .45 * np.clip((tsec - T_DRAW1) / 1.5, 0, 1))
    v8 = Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8))
    g1 = np.asarray(v8.filter(ImageFilter.GaussianBlur(4))).astype(np.float32) / 255
    g2 = np.asarray(v8.filter(ImageFilter.GaussianBlur(16))).astype(np.float32) / 255
    vv = np.clip(v * .9 + g1 * .5 + g2 * .6, 0, 1)
    red = tsec >= T_OPEN
    L = red_lut if red else lut
    rgb = L[(vv * 255).astype(np.uint8)]
    arr = np.maximum(bg, rgb)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    # ring of glyphs
    k = np.clip((tsec - 8.0) / 4, 0, 1) * (0.55 + 0.45 * np.clip((tsec - 15) / 10, 0, 1))
    if tsec >= T_DRAW1: k *= 1 - .5 * np.clip((tsec - T_DRAW1) / 1.5, 0, 1)
    if red: k = 1.0
    if k > 0: draw_ring(d, tsec, k, red)
    # pen tip
    if pen is not None and tsec < T_OPEN:
        x, y = pen
        for r_, a_ in ((18, 40), (9, 90), (4, 255)):
            d.ellipse([x - r_, y - r_, x + r_, y + r_], fill=(255, 235, 200, a_))
    # the eye opens
    if red:
        kk = np.clip((tsec - T_OPEN) / .35, 0, 1)
        pr = 30 * kk
        for r_, a_ in ((110 * kk, 40), (60 * kk, 80)):
            d.ellipse([CX - r_, CY - r_, CX + r_, CY + r_], fill=(255, 60, 40, a_))
        d.ellipse([CX - pr, CY - pr, CX + pr, CY + pr], fill=(10, 0, 0, 255))
        d.ellipse([CX - 7 * kk, CY - 16 * kk, CX + 7 * kk, CY + 16 * kk], fill=(255, 90, 60, 255))
    # captions + counter
    for a, b, text in CAPS:
        if a <= tsec < b and text:
            shown = text[:int((tsec - a) * 26)]
            y = 170 if "\n" in text else 200
            for ln in shown.split("\n"):
                d.text(((W - d.textlength(ln, font=f_cap)) / 2, y), ln, font=f_cap, fill=(238, 228, 214))
                y += 60
    if tsec < T_OPEN:
        tau_now = sim_time(tsec)
        ys0 = 1580
        for i in range(4):
            a = amp(i, tau_now) if T_DRAW0 <= tsec < T_DRAW1 else (0 if tsec >= T_DRAW1 else 1)
            label = f"P{i + 1}"
            x0 = 120 + i * 220
            d.text((x0, ys0), label, font=f_small, fill=(170, 120, 80))
            d.rectangle([x0 + 50, ys0 + 8, x0 + 50 + 130, ys0 + 22], outline=(90, 55, 35))
            d.rectangle([x0 + 50, ys0 + 8, x0 + 50 + int(130 * a), ys0 + 22], fill=(220, 140, 60))
    if tsec >= T_END:
        im = Image.new("RGB", (W, H), (5, 3, 3)); d = ImageDraw.Draw(im)
        kf = np.clip((tsec - T_END - .2) / .6, 0, 1)
        for j, ln in enumerate(["Made 100% with code.", "Every line. Every sound."]):
            d.text(((W - d.textlength(ln, font=f_end)) / 2, 900 + j * 54), ln, font=f_end,
                   fill=tuple(int(c * kf) for c in (215, 205, 195)))
    return im

nf = int(DUR * FPS)
checks = {int(x * FPS) for x in (2.0, 6.0, 12.0, 18.0, 24.5, 26.5, 28.5, 30.6, 33.0)}
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
       "-r", str(FPS), "-i", "-", "-i", os.path.join(OUT, "sigil.wav"), "-c:v", "libx264", "-preset", "veryfast",
       "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
       os.path.join(OUT, "harmonograph_sigil.mp4")]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fi in range(nf):
    im = render(fi)
    p.stdin.write(im.tobytes())
    if fi in checks:
        im.save(os.path.join(OUT, f"check_{fi:04d}.png"))
p.stdin.close(); p.wait()
print("done")
