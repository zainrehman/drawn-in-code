"""Shared helpers for the code-made Shorts: video writer, fonts, captions, voices, audio utils."""
import numpy as np, subprocess, os, glob
from scipy import signal
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SR, FPS, W, H = 44100, 30, 1080, 1920
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIFB = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SERIFI = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf" if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf") else SERIF
_fonts = {}
def font(path, size):
    k = (path, size)
    if k not in _fonts: _fonts[k] = ImageFont.truetype(path, size)
    return _fonts[k]

# ---------------------------------------------------------------- audio
def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], "band"); return signal.lfilter(b, a, x)
def lp(x, hi, order=2):
    b, a = signal.butter(order, hi / (SR / 2), "low"); return signal.lfilter(b, a, x)
def hp(x, lo, order=2):
    b, a = signal.butter(order, lo / (SR / 2), "high"); return signal.lfilter(b, a, x)
def norm(x):
    m = np.max(np.abs(x)); return x / m if m > 0 else x
def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf): return
    m = min(len(sig), len(buf) - i)
    if i < 0: sig = sig[-i:]; m = min(len(sig), len(buf)); i = 0
    buf[i:i + m] += sig[:m] * gain
def reverb(x, secs=1.2, decay=.3, seed=0):
    r = np.random.default_rng(seed); n = int(secs * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR / decay); ir[0] = 0
    return signal.fftconvolve(x, ir)[:len(x) + n] * .05
def boom(dur=3.0, f0=45, seed=1):
    r = np.random.default_rng(seed); n = int(dur * SR); tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * (f0 - 6 * tt) * tt) * np.exp(-tt * 1.6)
    s += lp(r.standard_normal(n), 250) * np.exp(-tt * 3) * 2.5
    return s
def say(text, args, outdir, name, speed_ratio=1.0):
    """espeak-ng -> float array at SR, trimmed. speed_ratio>1 = slower & lower."""
    p = os.path.join(outdir, name)
    subprocess.run(["espeak-ng", *args, "-w", p, text], check=True)
    sr, x = wavfile.read(p)
    x = x.astype(np.float64) / 32768
    up = int(round(SR * speed_ratio)); g = np.gcd(up, sr)
    x = signal.resample_poly(x, up // g, sr // g)
    nz = np.where(np.abs(x) > .01)[0]
    return norm(x[max(0, nz[0] - 200): nz[-1] + 1500])
def write_wav(path, L, R=None, drive=1.15):
    L = np.tanh(L * drive) * .9
    if R is None:
        wavfile.write(path, SR, (L * 32767).astype(np.int16))
    else:
        R = np.tanh(R * drive) * .9
        wavfile.write(path, SR, (np.stack([L, R], 1) * 32767).astype(np.int16))

# ---------------------------------------------------------------- video
class Writer:
    def __init__(self, outdir, name, wav, checks=()):
        self.outdir, self.checks = outdir, {int(round(c * FPS)) for c in checks}
        for f in glob.glob(os.path.join(outdir, "check_*.png")): os.remove(f)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
               os.path.join(outdir, name)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE); self.fi = 0
    def write(self, im):
        if not isinstance(im, Image.Image): im = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
        self.p.stdin.write(im.convert("RGB").tobytes())
        if self.fi in self.checks: im.save(os.path.join(self.outdir, f"check_{self.fi:04d}.png"))
        self.fi += 1
    def close(self):
        self.p.stdin.close(); self.p.wait()
        fs = sorted(glob.glob(os.path.join(self.outdir, "check_*.png")))
        if fs:
            ims = [Image.open(f).resize((270, 480)) for f in fs]
            cols = min(6, len(ims)); rows = (len(ims) + cols - 1) // cols
            c = Image.new("RGB", (270 * cols, 480 * rows))
            for k, im in enumerate(ims): c.paste(im, ((k % cols) * 270, (k // cols) * 480))
            c.save(os.path.join(self.outdir, "contact.png"))

def vignette(strength=.5, cy=H / 2, sx=.8, sy=.7):
    YY, XX = np.mgrid[0:H, 0:W]
    return (np.clip(1 - (((XX - W / 2) / (W * sx)) ** 2 + ((YY - cy) / (H * sy)) ** 2), 0, 1) ** strength)[..., None].astype(np.float32)

def captions(d, tsec, caps, fnt=None, color=(236, 230, 222), y1=170, y2=200, cps=26, lh=60, stroke=0):
    """caps: list of (start, end, text[, y]) ; typewriter reveal."""
    fnt = fnt or font(MONOB, 46)
    for c in caps:
        a, b, text = c[:3]
        if a <= tsec < b and text:
            shown = text[:int((tsec - a) * cps)]
            y = c[3] if len(c) > 3 and c[3] is not None else (y1 if "\n" in text else y2)
            col = c[4] if len(c) > 4 else color
            for ln in shown.split("\n"):
                d.text(((W - d.textlength(ln, font=fnt)) / 2, y), ln, font=fnt, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
                y += lh

def end_card(tsec, t0, lines=("Made 100% with code.", "Every pixel. Every sound."), bg=(4, 4, 5), col=(212, 210, 206)):
    im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
    k = float(np.clip((tsec - t0 - .2) / .6, 0, 1))
    f = font(MONO, 32)
    for j, ln in enumerate(lines):
        if ln:
            d.text(((W - d.textlength(ln, font=f)) / 2, 900 + j * 54), ln, font=f, fill=tuple(int(c * k) for c in col))
    return im

def glitch(arr, rng, n=14, shift=130):
    arr = np.array(arr).copy()
    for _ in range(n):
        y = rng.integers(0, H - 90); h = rng.integers(8, 90)
        arr[y:y + h] = np.roll(arr[y:y + h], rng.integers(-shift, shift), axis=1)
    arr[..., 0] = np.roll(arr[..., 0], 10, axis=1)
    return arr

def lut_from(stops):
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(np.linspace(0, 1, 256), [s[0] for s in stops], [s[1][c] for s in stops])
    return lut

def ascii_mask(rows, x0, y0, fnt, ch_h):
    """Render list of strings as an L mask (full frame)."""
    m = Image.new("L", (W, H), 0); d = ImageDraw.Draw(m)
    for r, line in enumerate(rows):
        d.text((x0, y0 + r * ch_h), line, font=fnt, fill=255)
    return np.asarray(m).astype(np.float32) / 255

def cell_color_image(col, cw, ch, gw, gh):
    """col: (rows, cols, 3) 0..1 -> (gh, gw, 3) pixel image via nearest cell lookup."""
    cx = np.minimum((np.arange(gw) / cw).astype(int), col.shape[1] - 1)
    cy = np.minimum((np.arange(gh) / ch).astype(int), col.shape[0] - 1)
    return col[cy][:, cx]
