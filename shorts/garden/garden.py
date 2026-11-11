"""Garden of Life - Conway's Game of Life, gently guided, grows a garden of
glyph flowers... which rot into a skull and go still. Every birth is a note."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from scipy.ndimage import gaussian_filter

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1970)
CS = 12
GC, GR = 88, 128
GX, GY = (W - GC * CS) // 2, 300
GPS = 10                                     # generations per second
T_GROW0, T_GROW1 = 3.0, 11.0
T_ROT0, T_ROT1 = 18.0, 23.0
T_FREEZE = 27.8
T_GLIDER = 28.6
T_CUT = 32.0
DUR = 35.0

# ------------------------------------------------------------------ masks
def mk(fn):
    im = Image.new("L", (GC * 4, GR * 4), 0); fn(ImageDraw.Draw(im))
    return np.asarray(im.resize((GC, GR), Image.BOX)) > 100
FLOWERS = [(20, 62, 9, 6, (235, 80, 120)), (46, 44, 11, 7, (240, 150, 60)), (70, 70, 8, 5, (180, 110, 240))]
def stems(d):
    for x, y, r, n, _ in FLOWERS:
        s = 4
        d.line([(x * s, GR * s), (x * s + 6, (y + 4) * s)], fill=255, width=9)
        d.ellipse([(x - 8) * s, (y + 30) * s, x * s, (y + 34) * s], fill=255)
        d.ellipse([x * s, (y + 44) * s, (x + 9) * s, (y + 48) * s], fill=255)
def petals(d):
    for x, y, r, n, _ in FLOWERS:
        s = 4
        for k in range(n):
            a = 2 * np.pi * k / n
            px, py = x + np.cos(a) * r * .9, y + np.sin(a) * r * .9
            d.ellipse([(px - r * .6) * s, (py - r * .6) * s, (px + r * .6) * s, (py + r * .6) * s], fill=255)
def centers(d):
    for x, y, r, n, _ in FLOWERS:
        s = 4; d.ellipse([(x - r * .38) * s, (y - r * .38) * s, (x + r * .38) * s, (y + r * .38) * s], fill=255)
M_STEM, M_PETAL, M_CENTER = mk(stems), mk(petals), mk(centers)
M_PETAL &= ~M_CENTER
M_FLOWER = M_STEM | M_PETAL | M_CENTER
def skull(d):
    s = 4; cx = GC / 2
    d.ellipse([(cx - 30) * s, 26 * s, (cx + 30) * s, 84 * s], fill=255)
    d.rectangle([(cx - 19) * s, 74 * s, (cx + 19) * s, 100 * s], fill=255)
def skull_holes(d):
    s = 4; cx = GC / 2
    for ex in (-13, 13): d.ellipse([(cx + ex - 9) * s, 50 * s, (cx + ex + 9) * s, 68 * s], fill=255)
    d.polygon([((cx - 4) * s, 72 * s), ((cx + 4) * s, 72 * s), (cx * s, 79 * s)], fill=255)
    for k in range(-15, 16, 6): d.rectangle([(cx + k) * s, 87 * s, (cx + k + 2) * s, 100 * s], fill=255)
M_SKULL = mk(skull) & ~mk(skull_holes)
dissolve = gaussian_filter(rng.random((GR, GC)), 1.5); dissolve = (dissolve - dissolve.min()) / (dissolve.max() - dissolve.min())
grow_noise = rng.random((GR, GC)) * 6
rows_idx = np.arange(GR)[:, None] * np.ones((1, GC))

def target(tsec):
    if tsec < T_GROW0: return None
    reveal = GR - (GR + 10) * np.clip((tsec - T_GROW0) / (T_GROW1 - T_GROW0), 0, 1)
    fl = M_FLOWER & (rows_idx + grow_noise > reveal)
    if tsec < T_ROT0: return fl
    k = np.clip((tsec - T_ROT0) / (T_ROT1 - T_ROT0), 0, 1)
    return (fl & (dissolve > k)) | (M_SKULL & (dissolve < k))

COLMAP = np.zeros((GR, GC, 3), np.float32) + np.array([110, 200, 110])
for (x, y, r, n, c) in FLOWERS:
    near = (np.hypot(np.arange(GC)[None] - x, np.arange(GR)[:, None] - y) < r * 1.7)
    COLMAP[near & M_PETAL] = c
COLMAP[M_CENTER] = (250, 220, 90)
BONE = np.array([215, 205, 180], np.float32)

# ------------------------------------------------------------------ simulate all generations up front
def nbr(a):
    a = a.astype(np.int8)
    return sum(np.roll(np.roll(a, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
GLIDER = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], bool)[::-1]     # travels up-right... flipped to go up
cells = np.zeros((GR, GC), bool)
cells[96:124, 30:58] = rng.random((28, 28)) < .45
gens = []; births_log = []
n_gens = int(DUR * GPS) + 2
glider_pos = None
for g in range(n_gens):
    tsec = g / GPS
    if tsec < T_FREEZE:
        n = nbr(cells); M = target(tsec)
        if M is None:
            new = (n == 3) | (cells & (n == 2))
        else:
            pin = rng.random((GR, GC))
            birth = (n == 3) | (M & (n >= 2) & (n <= 4) & (pin < .35)) | (M & (n <= 1) & (pin < .025))
            surv = cells & np.where(M, (n >= 1) & (n <= 5), ((n == 2) | (n == 3)) & (pin < .85))
            new = birth | surv
            new &= M | (pin < .97)
        b = new & ~cells
        births_log.append((int(b.sum()), float(np.mean(np.where(b)[1]) if b.any() else GC / 2)))
        cells = new
    else:
        births_log.append((0, 0))
        if tsec >= T_GLIDER:
            if glider_pos is None:
                glider_pos = [57, 31]                     # emerges from the left eye socket
                gl = np.zeros((GR, GC), bool)
            gl = np.zeros((GR, GC), bool)
            k = int((tsec - T_GLIDER) * GPS)
            phase = k % 4
            pats = [np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]), np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]]),
                    np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]), np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])]
            pat = pats[phase][::-1, ::-1].astype(bool)      # heading up-left
            r0, c0 = glider_pos[0] - k // 4, glider_pos[1] - k // 4
            if 0 <= r0 < GR - 3 and 0 <= c0 < GC - 3:
                gl[r0:r0 + 3, c0:c0 + 3] = pat
            gens.append(cells | gl); continue
    gens.append(cells.copy())

if os.environ.get('SIMONLY'):
    ims = [Image.fromarray((gens[int(x * GPS)] * 255).astype(np.uint8)).resize((176, 256)) for x in (2, 6, 10, 15, 20, 24, 27, 30)]
    c = Image.new('L', (176 * 8, 256)); [c.paste(im, (k * 176, 0)) for k, im in enumerate(ims)]; c.save('sim.png'); sys.exit()
# ------------------------------------------------------------------ audio: births become plucks
N = int(DUR * SR); t = np.arange(N) / SR
mix = np.zeros(N)
def pluck(f, dur=1.2, bright=.5):
    n = int(dur * SR); p = int(SR / f)
    buf = rng.uniform(-1, 1, p); out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = .996 * (buf[i % p] * bright + buf[(i + 1) % p] * (1 - bright)) if False else buf[i % p]
    return out
def ks(f, dur=1.3, decay=.994):
    n = int(dur * SR); p = max(2, int(SR / f))
    y = np.zeros(n + p); y[:p] = rng.uniform(-1, 1, p)
    for i in range(p, n + p, p):              # vectorised Karplus-Strong in period-sized blocks
        seg = y[i - p:i]
        y[i:i + p] = (decay * .5 * (seg + np.roll(seg, -1)))[:len(y[i:i + p])]
    return y[p:p + n] * np.exp(-np.arange(n) / SR * 2)
MAJ = [261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3, 784.0, 880.0]
MIN = [110.0, 116.5, 130.8, 146.8, 155.6, 174.6, 196.0, 207.7]
cache = {}
for g, (nb, mx) in enumerate(births_log):
    tsec = g / GPS
    if nb == 0 or tsec >= T_FREEZE: continue
    rot = np.clip((tsec - T_ROT0) / (T_ROT1 - T_ROT0), 0, 1)
    scale = MIN if rot > rng.random() else MAJ
    if g % 2 and nb < 20: continue
    f = scale[int(np.clip(mx / GC * len(scale), 0, len(scale) - 1))]
    if f not in cache: cache[f] = ks(f)
    add(mix, cache[f], tsec, min(.16, .02 + .025 * np.log1p(nb)))
pad = sum(np.sin(2 * np.pi * f * t) for f in (130.8, 196.0, 261.6)) * np.interp(t, [0, 3, 12, T_ROT0, T_ROT1, T_FREEZE, T_FREEZE + .05], [0, .02, .04, .04, 0, 0, 0])
dark = sum(np.sin(2 * np.pi * f * t) for f in (55, 58.3, 82.4)) * np.interp(t, [0, T_ROT0, T_ROT1, T_FREEZE, T_FREEZE + .05], [0, 0, .08, .1, 0])
mix += pad + dark
for k in range(int((T_CUT - T_GLIDER) * GPS / 2)):         # the glider ticks
    tick = np.sin(2 * np.pi * (1200 + 40 * k) * np.arange(int(.03 * SR)) / SR) * np.exp(-np.arange(int(.03 * SR)) / SR * 150)
    add(mix, tick, T_GLIDER + k * 2 / GPS, .12)
add(mix, boom(2.5, 40, 2), T_FREEZE, .6)
write_wav(os.path.join(OUT, "garden.wav"), mix)

# ------------------------------------------------------------------ glyph sprites + render
GLY = " ·+oO@"
fnt = font(MONOB, 14)
SPR = np.zeros((len(GLY), CS, CS), np.float32)
for i, ch in enumerate(GLY):
    im = Image.new("L", (CS, CS), 0); d = ImageDraw.Draw(im)
    w = d.textlength(ch, font=fnt); d.text(((CS - w) / 2, -3), ch, font=fnt, fill=255)
    SPR[i] = np.asarray(im) / 255
age = np.zeros((GR, GC), np.float32); ghost = np.zeros((GR, GC), np.float32)
vig = vignette(.45)
CAPS = [(.3, 4.2, "Four simple rules.\nNobody controls them."),
        (4.6, 10.6, "Left alone, the cells\ngrow a garden."),
        (12.4, 17.4, "Every time."),
        (18.2, 23.4, "But gardens\ndon't last."),
        (24.2, T_FREEZE + .4, "Eventually,\neverything stops."),
        (T_GLIDER + .2, T_CUT, "Almost everything.", None, (225, 70, 55))]
def render(fi):
    global age, ghost
    tsec = fi / FPS
    g = min(int(tsec * GPS), len(gens) - 1)
    c = gens[g]
    if fi % 3 == 0 and tsec < T_FREEZE:
        age = np.where(c, age + 1, 0); ghost = np.where(c, 1, ghost * .7)
    gi = np.where(c, np.clip(1 + (age // 3).astype(int), 1, 5), np.where(ghost > .25, 1, 0))
    tiles = SPR[gi]                                                   # (GR, GC, CS, CS)
    img = tiles.transpose(0, 2, 1, 3).reshape(GR * CS, GC * CS)
    rot = np.clip((tsec - T_ROT0) / (T_ROT1 - T_ROT0), 0, 1)
    col = COLMAP * (1 - rot) + BONE * rot
    if tsec >= T_FREEZE: col = col * (1 - .35 * np.clip((tsec - T_FREEZE) / 1, 0, 1))
    shade = np.where(c, 1.0, ghost * .4)[..., None]
    youth = np.where(c & (age < 3), 1.0, 0)[..., None]
    colc = np.clip(col * (.55 + .45 * shade) + youth * 60, 0, 255) * shade
    if tsec >= T_GLIDER:                                             # the escaping glider glows red
        moving = c & ~gens[int(T_FREEZE * GPS)]
        colc[moving] = (255, 70, 50)
    colimg = np.repeat(np.repeat(colc, CS, 0), CS, 1)
    layer = colimg * img[..., None]
    arr = np.zeros((H, W, 3), np.float32) + np.array([6, 8, 7])
    arr[GY:GY + GR * CS, GX:GX + GC * CS] += layer
    small = Image.fromarray(np.clip(arr[::4, ::4], 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR)
    arr = arr + np.asarray(small).astype(np.float32) * .8
    im = Image.fromarray(np.clip(arr * vig, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im, "RGBA")
    if tsec >= T_GLIDER:
        moving = c & ~gens[int(T_FREEZE * GPS)]
        if moving.any():
            ry, rx = np.where(moving); x = GX + rx.mean() * CS + 6; y = GY + ry.mean() * CS + 6
            for rr, a in ((70, 30), (40, 60), (22, 90)):
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 60, 40, a))
    d.text((60, 1900 - 120), f"GEN {g:04d}" if tsec < T_FREEZE else f"GEN {int(T_FREEZE * GPS):04d}  (still)", font=font(MONO, 26), fill=(110, 130, 115))
    captions(d, tsec, CAPS, font(MONOB, 46), y1=140, y2=170)
    return im

checks = [2, 6, 9, 13, 17, 20, 23, 26, 29.5, 31.5, 34]
wr = Writer(OUT, "garden_of_life.mp4", os.path.join(OUT, "garden.wav"), checks)
for fi in range(int(DUR * FPS)):
    tsec = fi / FPS
    wr.write(end_card(tsec, T_CUT + .1) if tsec >= T_CUT else render(fi))
wr.close()
print("done")
