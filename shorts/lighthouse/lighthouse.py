"""The Lighthouse - ASCII night sea. Each time the beam comes round it lights
the water, and the figures in it are closer. On the last turn it stops on you."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1871)
P = 5.5
T_F0 = 5.0                                # first time the beam points at us
FLARES = [T_F0 + k * P for k in range(5)]
T_STOP = FLARES[-1]
T_CUT = T_STOP + 1.6
DUR = T_CUT + 3.0

def theta(tsec):
    if tsec >= T_STOP: return np.pi / 2
    return np.pi / 2 + 2 * np.pi * (tsec - T_F0) / P
def stage(tsec):
    return int(np.clip(np.floor((tsec - T_F0 + P / 2) / P), 0, 4))

# ------------------------------------------------------------------ audio
N = int(DUR * SR); t = np.arange(N) / SR
mix = np.zeros(N)
sw = .5 + .5 * np.sin(2 * np.pi * t / 7.3) ** 2
waves = norm(lp(rng.standard_normal(N), 500)) * (.06 + .1 * sw)
crash = norm(bp(rng.standard_normal(N), 300, 3000)) * np.clip(np.sin(2 * np.pi * t / 7.3 - 1), 0, 1) ** 6 * .12
wind = norm(bp(rng.standard_normal(N), 400, 1200)) * (.03 + .02 * np.sin(2 * np.pi * .11 * t))
mix += (waves + crash + wind) * np.interp(t, [0, 1, T_CUT, T_CUT + .02, DUR], [0, 1, 1, 0, 0])
# beam whoosh each time it sweeps past us
th = np.array([theta(x) for x in np.arange(0, DUR, 1 / 200)])
f_env = np.interp(t, np.arange(0, DUR, 1 / 200), np.clip(np.sin(th), 0, 1) ** 8)
mix += norm(bp(rng.standard_normal(N), 150, 900)) * f_env * .16 * (t < T_CUT)
mix += np.sin(2 * np.pi * 60 * t) * f_env * .05 * (t < T_CUT)
def horn(at, dur=2.6, gain=.22):
    n = int(dur * SR); tt = np.arange(n) / SR
    env = np.interp(tt, [0, .25, dur - .5, dur], [0, 1, .9, 0])
    s = sum(np.sign(np.sin(2 * np.pi * f * tt)) * g for f, g in ((73.4, 1), (73.9, .8), (110, .35)))
    add(mix, lp(s, 600) * env * gain, at); add(mix, reverb(lp(s, 600) * env, 3, .9, int(at)) * gain, at, .6)
horn(2.0, 2.6, .07); horn(13.4, 2.2, .065); horn(T_STOP - .2, 3.0, .11)
for k, ft in enumerate(FLARES[1:-1]):                     # dread pulse on each reveal
    add(mix, boom(1.8, 50 - 3 * k, k), ft, .35 + .1 * k)
add(mix, boom(3.0, 36, 9), T_STOP, 1.0)
n = int(2 * SR); tt = np.arange(n) / SR
add(mix, norm(hp(rng.standard_normal(n), 2000)) * np.interp(tt, [0, 1.4, 1.6, 2], [0, .25, 0, 0]), T_STOP - .2)
drone = sum(np.sin(2 * np.pi * f * t) for f in (49, 49.4, 73.5)) * np.interp(t, [0, 6, T_STOP, T_CUT, T_CUT + .02], [0, .04, .1, .1, 0])
mix += drone
write_wav(os.path.join(OUT, "lighthouse.wav"), mix * .85)
if os.environ.get("AUDIOONLY"): sys.exit()

# ------------------------------------------------------------------ scene at cell resolution
COLS, ROWS = 76, 58
CW, CH = 13.25, 26
GW, GH = int(COLS * CW), ROWS * CH
GX, GY = (W - GW) // 2, 300
HZ = 22
LAMP = (5.0, 13.0)                       # row, col
R, C = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
dxs, dys = C - LAMP[1], (R - LAMP[0]) * 2.0     # cells are twice as tall
ang = np.arctan2(dys, dxs); dist = np.hypot(dxs, dys)
sea = R >= HZ
depth = np.clip((R - HZ) / (ROWS - HZ), 0, 1)
cliff = (C < 24 - (R - HZ) * .0) & (R > 12 + np.abs(C - 9) * .35) & (R < HZ + 3 + (C < 18) * 6)
tower = (np.abs(C - LAMP[1]) <= 2.2 + (R - 5) * .12) & (R >= 6) & (R <= 16)
stars = (rng.random((ROWS, COLS)) < .02) & (R < HZ - 2)

def figure_mask(cx, foot_row, h, wide=2.0):
    """A standing figure in the water: head + shoulders above waterline."""
    m = np.zeros((ROWS, COLS), np.float32)
    top = foot_row - h
    hr = max(.6, h * .16)
    head = ((C - cx) / (hr * 1.1 * wide)) ** 2 + ((R - (top + hr)) / (hr * .9)) ** 2 <= 1
    body = (R > top + hr * 1.6) & (R <= foot_row) & (np.abs(C - cx) <= h * .28 * wide * np.clip((R - top) / (h * .5), .5, 1))
    m[head | body] = 1
    return m
STAGES = [
    [],
    [(40, HZ + 3, 3)],
    [(45, HZ + 11, 7), (29, HZ + 4, 3), (57, HZ + 4, 3)],
    [(40, HZ + 26, 15), (21, HZ + 15, 8), (60, HZ + 14, 8), (32, HZ + 6, 4), (50, HZ + 5, 3), (68, HZ + 6, 4)],
]
FIGS = []
for st in STAGES:
    m = np.zeros((ROWS, COLS), np.float32)
    for f in st: m = np.maximum(m, figure_mask(*f))
    FIGS.append(m)
giant = figure_mask(38, ROWS + 18, 52, wide=2.2)
FIGS.append(giant)
GIANT_EYES = [(int(ROWS - 30), 31), (int(ROWS - 30), 45)]

SKYRAMP = np.array(list(" .:-=+*#%@"))
SEARAMP = np.array(list(" .-~~≈=*#@"))
def frame_cells(tsec):
    th = theta(tsec)
    toward = max(0.0, np.sin(th)) ** 6
    phi = 0.0 if np.cos(th) >= 0 else np.pi
    side = abs(np.cos(th))
    w = .05 + .03 * (1 - side)
    da = np.angle(np.exp(1j * (ang - phi)))
    beam = np.exp(-(da / w) ** 2) * side * np.exp(-dist / 140) * (dist > 2)
    # sky
    v = np.where(sea, 0, .05 + .07 * (1 - R / HZ))
    v = v + stars * .25
    # sea waves
    ph = tsec * 1.2
    wave = .5 + .5 * np.sin(C * (.9 - .6 * depth) + ph + R * 1.7) * np.sin(C * .23 - ph * .6 + R * .5)
    v = np.where(sea, .08 + .1 * wave * (.3 + depth), v)
    # beam on the water: a lit band under the beam near the horizon, a broad wash when it faces us
    sea_lit = np.exp(-(da / (w * 2.5)) ** 2) * side * np.exp(-np.abs(R - HZ - 1) / 4) * .9
    sea_lit = sea_lit + toward * (.45 + .75 * depth) * np.exp(-((C - 40) / (20 + 30 * depth)) ** 2)
    v = v + beam * 1.4 * (~sea) + np.where(sea, sea_lit * (.6 + .5 * wave), 0)
    # land and tower
    v = np.where(cliff, .03, v)
    v = np.where(tower, .32 + .1 * (C < LAMP[1]), v)
    v = np.where((np.abs(R - LAMP[0]) < 1.2) & (np.abs(C - LAMP[1]) < 2), .6 + .4 * max(toward, .2), v)
    # figures (always dark - visible only against lit water)
    st = stage(tsec)
    fig = FIGS[st] if not (st == 4 and tsec < T_STOP - .05) else FIGS[3]
    if st == 4 and tsec >= T_STOP - .05: fig = FIGS[4]
    v = v * (1 - fig) + fig * .0
    warm = np.clip(beam * 2 + sea_lit * 1.5, 0, 1)
    return np.clip(v, 0, 1), warm, fig, toward, sea

FONT = font(MONO, 22)
vig = vignette(.4)
CAPS = [(.3, 4.6, "The light comes round\nevery few seconds."),
        (5.6, 10.0, "Something is out there\nbetween the flashes."),
        (11.2, 15.6, "Closer every time."),
        (16.6, 21.0, "There are more\nof them now."),
        (22.2, T_STOP - .3, "Then the light\nstopped turning.")]

def render(fi):
    tsec = fi / FPS
    v, warm, fig, toward, seam = frame_cells(tsec)
    idx = np.clip((v * 10).astype(int), 0, 9)
    chars = np.where(seam, SEARAMP[idx], SKYRAMP[idx])
    chars = np.where(fig > .5, " ", chars)
    col = np.stack([.42 + .58 * warm, .52 + .40 * warm, .70 - .05 * warm], -1) * (.45 + .75 * v[..., None])
    rows = ["".join(r) for r in chars]
    m = ascii_mask(rows, GX, GY, FONT, CH)
    colimg = cell_color_image(np.clip(col, 0, 1.2), CW, CH, GW, GH) * 255
    arr = np.zeros((H, W, 3), np.float32) + np.array([4, 6, 10])
    mm = m[GY:GY + GH, GX:GX + GW, None]
    arr[GY:GY + GH, GX:GX + GW] = arr[GY:GY + GH, GX:GX + GW] * (1 - mm) + colimg * mm
    # lamp glow and the flare when it faces us
    lx, ly = GX + LAMP[1] * CW, GY + LAMP[0] * CH + 12
    th = theta(tsec)
    YY, XX = np.ogrid[0:H, 0:W]
    rr = np.sqrt((XX - lx) ** 2 + (YY - ly) ** 2)
    glow = np.exp(-rr / (26 + 110 * toward)) * (.3 + 1.1 * toward)
    if tsec >= T_STOP:
        k = np.clip((tsec - T_STOP) / 1.2, 0, 1)
        glow = glow + k * .9 * np.exp(-rr / 900)
    arr = arr + glow[..., None] * np.array([255, 236, 190])
    arr = arr * vig
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    if tsec >= T_STOP:                            # eyes of the one in front of you
        k = np.clip((tsec - T_STOP - .3) / .3, 0, 1)
        for r, c in GIANT_EYES:
            x, y = GX + c * CW, GY + r * CH
            d.ellipse([x - 9 * k, y - 6 * k, x + 9 * k, y + 6 * k], fill=(255, 250, 235))
    captions(d, tsec, CAPS, font(MONOB, 46), y1=130, y2=160, stroke=4)
    if T_CUT - .3 <= tsec < T_CUT:
        return glitch(np.asarray(im), rng)
    return im

checks = [2.5, FLARES[1], FLARES[1] + 2.75, FLARES[2], FLARES[3], FLARES[3] + 1.5, T_STOP + .2, T_STOP + 1.2, DUR - 1]
wr = Writer(OUT, "the_lighthouse.mp4", os.path.join(OUT, "lighthouse.wav"), checks)
for fi in range(int(DUR * FPS)):
    tsec = fi / FPS
    wr.write(end_card(tsec, T_CUT + .1) if tsec >= T_CUT else render(fi))
wr.close()
print("done", DUR)
