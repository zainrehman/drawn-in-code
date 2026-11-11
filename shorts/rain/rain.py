"""Rain on Glass - an ASCII rainy window. Drops clear the fog, and every
lightning flash shows the figure outside a little closer.
Everything (image, voice, sound) is generated here. Needs espeak-ng + ffmpeg.
Output: rain_on_glass.mp4 (1080x1920, 30 fps, stereo)
"""
import numpy as np, subprocess, os
from scipy import signal
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(72)
SR, FPS, W, H = 44100, 30, 1080, 1920
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

LIGHTNING = [10.4, 17.0, 25.4]
T_HANDS = 22.0
T_TAPS = [26.45, 26.7]
T_BLACK = 27.2
DUR = 34.0

# ======================================================================= audio
N = int(DUR * SR); t = np.arange(N) / SR
L = np.zeros(N); R = np.zeros(N)
def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band"); return signal.lfilter(b, a, x)
def lp(x, hi, order=2):
    b, a = signal.butter(order, hi / (SR / 2), "low"); return signal.lfilter(b, a, x)
rain_env = np.interp(t, [0, 1.2, T_BLACK, T_BLACK + .05, 30.5, 32.5, DUR], [0, 1, 1, .55, .55, 0, 0])
for ch in (L, R):
    n = rng.standard_normal(N)
    hiss = bp(n, 700, 9000); hiss /= np.max(np.abs(hiss))
    body = lp(rng.standard_normal(N), 400); body /= np.max(np.abs(body))
    drips = np.zeros(N); idx = rng.integers(0, N, int(DUR * 220)); drips[idx] = rng.standard_normal(len(idx))
    drips = bp(drips, 1500, 7000); drips /= np.max(np.abs(drips))
    taps = np.zeros(N); idx = rng.integers(0, N, int(DUR * 9)); taps[idx] = rng.uniform(.4, 1, len(idx))
    taps = bp(taps, 2500, 6000, 1); taps /= np.max(np.abs(taps))
    ch += (hiss * .18 + body * .1 + drips * .1 + taps * .22) * rain_env
drone = sum(np.sin(2 * np.pi * f * t) * g for f, g in [(49, 1), (49.3, .8), (73.6, .4), (98.2, .2)])
drone *= np.interp(t, [0, 4, 9, T_BLACK, T_BLACK + .02, DUR], [0, .03, .07, .1, 0, 0])
L += drone; R += drone
def thunder(at, gain=1.0, dur=4.0):
    i = int(at * SR); n = int(dur * SR); tt = np.arange(n) / SR
    crack = bp(rng.standard_normal(n), 800, 6000) * np.exp(-tt * 9) * .5
    rum = lp(rng.standard_normal(n), 120, 4); rum /= np.max(np.abs(rum))
    rum *= np.interp(tt, [0, .15, .6, dur], [0, 1, .7, 0]) * (1 + .4 * np.sin(2 * np.pi * 3.1 * tt))
    s = (crack + rum * 1.1) * gain
    m = min(n, N - i)
    L[i:i + m] += s[:m] * 1.0; R[i:i + m] += np.roll(s, 300)[:m] * .9
for k, lt in enumerate(LIGHTNING):
    thunder(lt + .25, gain=.55 + .2 * k)
# whisper "let me in", panned left -> right, then close and centred
W1 = os.path.join(OUT, "w1.wav")
if not os.path.exists(W1):                       # espeak-ng whisper, 22.05 kHz
    subprocess.run(["espeak-ng", "-v", "en+whisper", "-s", "120", "-p", "35", "-w", W1, "let me in"], check=True)
sr0, wv = wavfile.read(W1)
wv = signal.resample_poly(wv.astype(np.float64) / 32768, 441 * 115, 220 * 100)   # slower, lower
wv /= np.max(np.abs(wv))
ir = rng.standard_normal(int(.9 * SR)) * np.exp(-np.arange(int(.9 * SR)) / SR / .25)
def place(sig, at, pan, gain, wet):
    s = sig * (1 - wet) + signal.fftconvolve(sig, ir)[:len(sig) + int(.9 * SR)][:len(sig)] * wet * .08
    i = int(at * SR); m = min(len(s), N - i)
    pl = np.cos((pan + 1) * np.pi / 4); pr = np.sin((pan + 1) * np.pi / 4)
    L[i:i + m] += s[:m] * pl * gain; R[i:i + m] += s[:m] * pr * gain
place(wv, 19.6, -.85, .45, .6)
place(wv, 23.0, .8, .55, .4)
place(wv, 25.9, 0.0, .9, .05)
for at in T_TAPS:                                 # two knuckle taps on the glass
    i = int(at * SR); n = int(.12 * SR); tt = np.arange(n) / SR
    s = (np.sin(2 * np.pi * 1150 * tt) * .6 + np.sin(2 * np.pi * 2310 * tt) * .3) * np.exp(-tt * 55)
    s += rng.standard_normal(n) * np.exp(-tt * 400) * .5
    L[i:i + n] += s * .8; R[i:i + n] += s * .8
mixL = np.tanh(L * 1.2) * .9; mixR = np.tanh(R * 1.2) * .9
wavfile.write(os.path.join(OUT, "rain.wav"), SR, (np.stack([mixL, mixR], 1) * 32767).astype(np.int16))

# ======================================================================= scene
COLS, ROWS = 80, 54
CW, CH = 12.04, 24
GX, GY = int((W - COLS * CW) / 2), 330
SC = 6                                    # canvas px per cell (x); y is 2x
cvw, cvh = COLS * SC, ROWS * SC * 2
HORIZON = 430

def canvas(fn, blur=0):
    im = Image.new("L", (cvw, cvh), 0); fn(ImageDraw.Draw(im))
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im).astype(np.float32) / 255
def cells(a):
    return a.reshape(ROWS, SC * 2, COLS, SC).mean(axis=(1, 3))

yy = np.linspace(0, 1, cvh)[:, None] * np.ones((1, cvw))
sky = np.where(np.arange(cvh)[:, None] < HORIZON, .2 + .14 * (1 - yy), 0) * np.ones((1, cvw))
ground = np.where(np.arange(cvh)[:, None] >= HORIZON, .13, 0) * np.ones((1, cvw))
trees = canvas(lambda d: [d.ellipse([x - 30, HORIZON - 40 - (x * 37 % 50), x + 30, HORIZON + 20], fill=255)
                          for x in range(0, cvw + 40, 34)], 2)
lamp_pole = canvas(lambda d: (d.rectangle([398, 180, 402, HORIZON + 10], fill=255),
                              d.line([(400, 182), (382, 182)], fill=255, width=4)))
lamp_head = canvas(lambda d: d.ellipse([374, 176, 392, 192], fill=255), 1)
cone = canvas(lambda d: d.polygon([(376, 188), (390, 188), (470, HORIZON + 30), (300, HORIZON + 30)], fill=255), 10)
pool = canvas(lambda d: d.ellipse([280, HORIZON - 4, 480, HORIZON + 40], fill=255), 10)
reflect = canvas(lambda d: d.rectangle([376, HORIZON + 30, 390, cvh], fill=255), 6)
halo = canvas(lambda d: d.ellipse([340, 140, 426, 228], fill=255), 22)
scene_v = np.clip(sky * (1 - trees) + trees * .03 + ground + cone * .45 + pool * .5 + reflect * .28 * (0.6 + 0.4 * np.sin(np.arange(cvh) / 9)[:, None])
                  + halo * .5 + lamp_head * 1.0 - lamp_pole * .2, 0, 1)
warm = np.clip(cone + pool + reflect * .8 + halo + lamp_head, 0, 1)

def figure(stage):
    def fn(d):
        if stage == 0:      # far, under the lamp
            cx, top, h = 352, 380, 70
        elif stage == 1:
            cx, top, h = 240, 290, 230
        else:
            cx, top, h = 240, 150, 600
        if stage <= 2:
            hr = h * .085
            d.ellipse([cx - hr, top, cx + hr, top + hr * 2.3], fill=255)
            sh = h * .16
            d.polygon([(cx - sh, top + hr * 2.6), (cx + sh, top + hr * 2.6), (cx + sh * 1.05, top + h * .62),
                       (cx + sh * .55, top + h), (cx - sh * .55, top + h), (cx - sh * 1.05, top + h * .62)], fill=255)
        else:               # pressed against the glass
            d.ellipse([125, 140, 355, 500], fill=255)
            d.polygon([(0, 640), (480, 640), (480, 560), (380, 470), (100, 470), (0, 560)], fill=255)
    return cells(canvas(fn, 1))
FIG = [figure(s) for s in range(4)]
EYES = [(np.array([203, 277]) / SC, 297 / (SC * 2))]   # stage-3 eyes in cell coords
FACE = cells(canvas(lambda d: (d.ellipse([150, 170, 330, 470], fill=255),), 14)) * (1 - cells(canvas(lambda d: (
    d.ellipse([175, 270, 230, 325], fill=255), d.ellipse([250, 270, 305, 325], fill=255),
    d.ellipse([210, 380, 270, 440], fill=255)), 4)))
hands = cells(canvas(lambda d: [
    (d.ellipse([cx - 34, 300, cx + 34, 390], fill=255),
     [d.rounded_rectangle([cx - 34 + k * 17, 240 - (k in (1, 2)) * 18, cx - 22 + k * 17, 320], 6, fill=255) for k in range(4)],
     d.rounded_rectangle([cx + (38 if cx > 240 else -50), 320, cx + (50 if cx > 240 else -38), 370], 6, fill=255))
    for cx in (60, 420)], 1))

scene_c = cells(scene_v); warm_c = cells(warm)
fog_glow = cells(np.clip(.22 + halo * .35 + cone * .15, 0, 1))
yy_c, xx_c = np.mgrid[0:ROWS, 0:COLS]

# ======================================================================= glass state (simulated ahead of time)
nf = int(DUR * FPS)
F = np.ones((ROWS, COLS)) * .95
drops = []
beads = {(int(r), int(c)): rng.choice(list(".'°,")) for r, c in zip(rng.integers(0, ROWS, 260), rng.integers(0, COLS, 260))}
streaks = []
pool_cols = set()
def step_glass(fi):
    tsec = fi / FPS
    global F
    F = np.minimum(1, F + .0016 + rng.random(F.shape) * .0012)
    rate = 1.4 + tsec * .06
    for _ in range(rng.poisson(rate * 3 / FPS * 3)):
        drops.append(dict(x=rng.uniform(1, COLS - 2), y=rng.uniform(-2, ROWS * .6), v=rng.uniform(6, 15),
                          big=rng.random() < .35, pause=0))
    heads = {}
    for d in drops:
        if d["pause"] > 0:
            d["pause"] -= 1
        else:
            dy = d["v"] / FPS
            d["y"] += dy; d["x"] += rng.normal(0, .25) * dy
            if rng.random() < .03: d["pause"] = rng.integers(3, 15)
        r, c = int(d["y"]), int(np.clip(d["x"], 0, COLS - 1))
        if 0 <= r < ROWS:
            F[r, c] = 0
            if d["big"]:
                F[r, max(0, c - 1)] *= .3; F[r, min(COLS - 1, c + 1)] *= .3
                F[max(0, r - 1), c] = 0
            heads[(r, c)] = "O" if d["big"] else "o"
            beads.pop((r, c), None)
        if d["y"] >= ROWS - 2:
            pool_cols.add(c)
    drops[:] = [d for d in drops if d["y"] < ROWS - 1.5]
    if rng.random() < .5:
        beads[(int(rng.integers(0, ROWS)), int(rng.integers(0, COLS)))] = rng.choice(list(".'°,"))
    for _ in range(rng.poisson(7)):
        streaks.append([rng.uniform(-10, COLS), rng.uniform(-5, ROWS * .3)])
    for s in streaks:
        s[0] += .5; s[1] += 1.6
    streaks[:] = [s for s in streaks if s[1] < ROWS]
    return heads

def flash_at(tsec):
    v = 0
    for lt in LIGHTNING:
        dt = tsec - lt
        if 0 <= dt < .07: v = max(v, 1)
        elif .07 <= dt < .14: v = max(v, .12)
        elif .14 <= dt < 1.2: v = max(v, .95 * np.exp(-(dt - .14) * 4.5))
    return v

def stage_at(tsec):
    return sum(tsec >= lt + .02 for lt in LIGHTNING)

RAMP = " .:-=+*#%@"
FOGRAMP = " .,:;~"
POOL_TEXT = "LET ME IN"
pool_start = (COLS - len(POOL_TEXT) * 2) // 2

def ascii_frame(fi, heads):
    tsec = fi / FPS
    fl = flash_at(tsec); st = stage_at(tsec)
    fig = FIG[st]
    v = scene_c * (1 - fl) + (np.where(yy_c < HORIZON / (SC * 2), .95, .35)) * fl
    v = v * (1 - fig) + fig * (.0 if st < 3 else .0)
    if st == 3:
        v = v + FACE * (.62 + .3 * fl)
    col_scene = np.stack([.45 + .55 * warm_c, .55 + .25 * warm_c, .85 - .4 * warm_c], -1)
    col_scene = col_scene * (1 - fl) + np.array([.85, .85, 1.0]) * fl
    hand_k = np.clip((tsec - T_HANDS) / 2.5, 0, 1) if tsec < T_BLACK else 0
    Fv = F * (1 - .65 * fl)
    if hand_k:
        Fv = Fv * (1 - hands * hand_k)
        v = v * (1 - hands * hand_k) + hands * hand_k * .8
    vis = v * (1 - Fv * .5) + Fv * fog_glow * .55 * (1 + 2 * fl)
    chars = np.empty((ROWS, COLS), dtype="<U1")
    fogged = Fv > .7
    ri = np.clip((vis * len(RAMP)).astype(int), 0, len(RAMP) - 1)
    fi_ = np.clip((vis * 2.2 * len(FOGRAMP)).astype(int), 0, len(FOGRAMP) - 1)
    chars[:] = np.array(list(RAMP))[ri]
    chars[fogged] = np.array(list(FOGRAMP))[fi_][fogged]
    for sx, sy in streaks:                         # rain outside
        for k in range(3):
            r, c = int(sy - k * 1), int(sx - k * .35)
            if 0 <= r < ROWS and 0 <= c < COLS and Fv[r, c] < .5 and fig[r, c] < .5:
                chars[r, c] = "/"; vis[r, c] = max(vis[r, c], .45)
    if fl > .3 and st < 3:
        chars[fig > .5] = " "
    bright = np.clip(.35 + vis * .9, 0, 1.2)
    col = col_scene * (1 - Fv[..., None]) + np.array([.62, .68, .78])[None, None] * Fv[..., None]
    col = col * bright[..., None]
    for (r, c), ch in beads.items():
        if fogged[r, c]:
            chars[r, c] = ch; col[r, c] = [.75, .82, .92]
    for (r, c), ch in heads.items():
        chars[r, c] = ch; col[r, c] = [.95, .98, 1.0]
    if st == 3 and tsec < T_BLACK:                 # eyes
        for ex in EYES[0][0]:
            r, c = int(EYES[0][1]), int(ex)
            chars[r, c - 1:c + 2] = ["(", "@", ")"]
            col[r, c - 1:c + 2] = [1.0, .3, .25]
            chars[r - 1, c - 1:c + 2] = ["_", "_", "_"]; col[r - 1, c - 1:c + 2] = [.6, .6, .65]
    # letters pooling on the sill
    pr = ROWS - 1
    for c in pool_cols:
        if c < COLS: chars[pr, c] = "~"; col[pr, c] = [.55, .65, .8]
    if False:
        k = int(np.clip((tsec - 18.5) / 5.0, 0, 1) * len(POOL_TEXT))
        for j, ch in enumerate(POOL_TEXT[:k]):
            c = pool_start + j * 2
            if ch != " ":
                chars[pr, c] = ch; col[pr, c] = [.95, .35, .3]
            else:
                chars[pr, c] = " "
    return chars, np.clip(col, 0, 1)

# ======================================================================= render
f_ascii = ImageFont.truetype(MONO, 20)
f_cap = ImageFont.truetype(MONOB, 44)
f_end = ImageFont.truetype(MONO, 32)
f_pool = ImageFont.truetype(MONOB, 46)
CAPS = [(0.3, 4.5, "It's been raining\nfor three days."),
        (4.8, 10.2, "Someone has been standing\nout there the whole time."),
        (11.0, 16.8, "Every time the\nlightning comes..."),
        (17.4, 24.2, "...they're closer."),
        (T_BLACK + .4, 30.4, "Don't open the window.")]
GWp, GHp = int(COLS * CW), ROWS * CH
frame_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
fd = ImageDraw.Draw(frame_layer)
wood, wood_hi = (34, 24, 18, 255), (70, 50, 36, 255)
fd.rectangle([GX - 34, GY - 34, GX + GWp + 34, GY + GHp + 46], outline=wood, width=34)
fd.rectangle([GX - 2, GY - 2, GX + GWp + 2, GY + GHp + 2], outline=wood_hi, width=2)
mx, my = -100, GY + int(GHp * .13)
fd.rectangle([mx - 9, GY, mx + 9, GY + GHp], fill=wood); fd.line([(mx - 9, GY), (mx - 9, GY + GHp)], fill=wood_hi, width=2)
fd.rectangle([GX, my - 9, GX + GWp, my + 9], fill=wood); fd.line([(GX, my - 9), (GX + GWp, my - 9)], fill=wood_hi, width=2)
fd.rectangle([GX - 60, GY + GHp + 46, GX + GWp + 60, GY + GHp + 120], fill=(46, 32, 24, 255))   # sill
YY, XX = np.mgrid[0:H, 0:W]
vig = (np.clip(1 - (((XX - W / 2) / (W * .8)) ** 2 + ((YY - H / 2) / (H * .7)) ** 2), 0, 1) ** .45)[..., None]

def caption(d, tsec, color=(234, 230, 222)):
    for a, b, text in CAPS:
        if a <= tsec < b:
            shown = text[:int((tsec - a) * 26)]
            y = 140 if "\n" in text else 168
            if b > T_BLACK + .3: y = 900
            for ln in shown.split("\n"):
                d.text(((W - d.textlength(text.split("\n")[0] if False else ln, font=f_cap)) / 2, y), ln, font=f_cap, fill=color)
                y += 58

def render(fi, heads):
    tsec = fi / FPS
    if tsec >= T_BLACK:
        im = Image.new("RGB", (W, H), (3, 3, 5)); d = ImageDraw.Draw(im)
        caption(d, tsec, (225, 70, 60))
        if tsec >= 30.8:
            k = min(1, (tsec - 30.8) / .6)
            for j, ln in enumerate(["Made 100% with code.", "Every pixel. Every sound."]):
                d.text(((W - d.textlength(ln, font=f_end)) / 2, 900 + j * 54), ln, font=f_end,
                       fill=tuple(int(c * k) for c in (210, 210, 215)))
        return im
    chars, col = ascii_frame(fi, heads)
    mask = Image.new("L", (W, H), 0); md = ImageDraw.Draw(mask)
    for r in range(ROWS):
        md.text((GX, GY + r * CH), "".join(chars[r]), font=f_ascii, fill=255)
    colimg = np.repeat(np.repeat(col, CH, 0), int(np.ceil(CW)), 1)
    cx = (np.arange(GWp) / CW).astype(int)
    colimg = (col[:, cx].repeat(CH, 0) * 255)
    arr = np.zeros((H, W, 3), np.float32) + np.array([5, 6, 9])
    m = np.asarray(mask).astype(np.float32)[GY:GY + GHp, GX:GX + GWp, None] / 255
    arr[GY:GY + GHp, GX:GX + GWp] = arr[GY:GY + GHp, GX:GX + GWp] * (1 - m) + colimg * m
    fl = flash_at(tsec)
    if fl > .05:                                     # lightning lights the room too
        arr += fl * 40
    arr = np.clip(arr * vig, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.paste(frame_layer, (0, 0), frame_layer)
    if fl > .05:
        im = Image.blend(im, Image.new("RGB", (W, H), (200, 200, 230)), fl * .12)
    d = ImageDraw.Draw(im)
    caption(d, tsec)
    if tsec > 18.5:
        k = int(np.clip((tsec - 18.5) / 5.0, 0, 1) * len(POOL_TEXT) + .999)
        txt = POOL_TEXT[:k]
        x0 = (W - d.textlength(POOL_TEXT, font=f_pool)) / 2
        d.text((x0, GY + GHp + 56), txt, font=f_pool, fill=(205, 45, 38))
        for j, ch in enumerate(txt):          # drips under each letter
            if ch != " ":
                lx = x0 + d.textlength(POOL_TEXT[:j], font=f_pool) + 14
                ln = min(60, (tsec - 18.5 - j * 5 / len(POOL_TEXT)) * 14)
                if ln > 0: d.line([(lx, GY + GHp + 104), (lx, GY + GHp + 104 + ln)], fill=(150, 30, 28), width=3)
    return im

checks = {int(x * FPS) for x in (3.0, 8.5, 10.42, 14.0, 17.02, 21.0, 24.0, 25.42, 26.4, 28.5, 32.0)}
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
       "-r", str(FPS), "-i", "-", "-i", os.path.join(OUT, "rain.wav"), "-c:v", "libx264", "-preset", "veryfast",
       "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
       os.path.join(OUT, "rain_on_glass.mp4")]
PREVIEW = os.environ.get('PREVIEW')
p = None if PREVIEW else subprocess.Popen(cmd, stdin=subprocess.PIPE)
for fi in range(nf):
    heads = step_glass(fi)
    if PREVIEW and fi not in checks: continue
    im = render(fi, heads)
    if p: p.stdin.write(im.tobytes())
    if fi in checks:
        im.save(os.path.join(OUT, f"check_{fi:04d}.png"))
if p: p.stdin.close(); p.wait()
print("done")
