"""Ink in Water - one drop of glyph-ink blooms in still water, and for a few
seconds it forms a face. Fluid motion: vortex ring + curl noise, semi-Lagrangian advection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from scipy.ndimage import map_coordinates, gaussian_filter

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(8)
VSIGN = 1.0
T_DROP, T_HIT = 1.3, 1.75
T_FACE0, T_FACE1, T_FACE2, T_FACE3 = 15.0, 17.0, 18.8, 20.6
T_FLOOD = 23.5
T_BLACK = 27.6
DUR = 31.0

# ------------------------------------------------------------------ audio
N = int(DUR * SR); t = np.arange(N) / SR
mix = np.zeros(N)
amb = lp(rng.standard_normal(N), 260); amb = norm(amb) * (1 + .4 * np.sin(2 * np.pi * .13 * t))
mix += amb * np.interp(t, [0, 1, T_BLACK, T_BLACK + .02, DUR], [0, .09, .12, 0, 0])
def plop(seed, gain=1):
    r = np.random.default_rng(seed); n = int(.5 * SR); tt = np.arange(n) / SR
    f = 380 + 1100 * (1 - np.exp(-tt * 40))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 22)
    s += bp(r.standard_normal(n), 800, 5000) * np.exp(-tt * 60) * .4
    return s * gain
add(mix, plop(1), T_HIT, .7)
for _ in range(60):                                  # bubbles
    tm = rng.uniform(T_HIT + .2, T_BLACK - .5); n = int(.06 * SR); tt = np.arange(n) / SR
    f = rng.uniform(700, 2200)
    add(mix, np.sin(2 * np.pi * f * (1 + 3 * tt) * tt) * np.exp(-tt * 70), tm, rng.uniform(.03, .09))
# glassy tones, slow tremolo, pentatonic
for i, f in enumerate([392, 466.2, 523.3, 587.3, 698.5, 784]):
    st = T_HIT + 1.2 + i * 2.1
    n = int(5 * SR); tt = np.arange(n) / SR
    tone = np.sin(2 * np.pi * f * tt) * (1 + .3 * np.sin(2 * np.pi * 5 * tt)) * np.interp(tt, [0, .05, 5], [0, 1, 0]) ** 2
    if st < T_FACE0 - 1: add(mix, tone, st, .05)
# face: low choir swell + heartbeat
n = int((T_FACE3 - T_FACE0 + 1.5) * SR); tt = np.arange(n) / SR
choir = sum(np.sin(2 * np.pi * f * tt + k) * (1 + .2 * np.sin(2 * np.pi * (.2 + .05 * k) * tt))
            for k, f in enumerate([110, 110.6, 130.8, 164.8, 165.4, 220.2]))
add(mix, choir * np.interp(tt, [0, 2, 3.8, 5.6, n / SR], [0, .06, .07, .02, 0]), T_FACE0)
for b in np.arange(T_FACE1, T_FACE3, .85):
    for off, g in ((0, .45), (.22, .3)):
        add(mix, boom(.4, 40, int(b * 10)), b + off, g)
# flood: deep rising drone, then a second drop in the dark
n = int((T_BLACK - T_FLOOD) * SR); tt = np.arange(n) / SR
add(mix, sum(np.sin(2 * np.pi * f * tt) for f in (55, 58.3, 82.4)) * np.linspace(0, .12, n), T_FLOOD)
add(mix, plop(2), T_BLACK + .7, .9)
add(mix, reverb(plop(2), 2.0, .6), T_BLACK + .7, .5)
write_wav(os.path.join(OUT, "ink.wav"), mix)

# ------------------------------------------------------------------ simulation grid
COLS, ROWS = 76, 58
CW, CH = 13.25, 26
GW, GH = int(COLS * CW), ROWS * CH
GX, GY = (W - GW) // 2, 330
SH, SW = 174, 116                         # sim grid (roughly square cells)
D = np.zeros((SH, SW), np.float32)
yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)

def face_target():
    S = 4; im = Image.new("L", (SW * S, SH * S), 0); d = ImageDraw.Draw(im)
    cx, cy = SW * S / 2, SH * S * .47
    for k in range(70):                    # flowing hair strands
        side = -1 if k % 2 else 1
        x0 = cx + side * rng.uniform(10, 46) * S / 4 * 4
        pts = []
        for j in range(60):
            s = j / 59
            x = x0 + side * (s * 46 * S) + np.sin(s * 6 + k) * 10 * S
            y = cy - 50 * S + s * 120 * S
            pts.append((x, y))
        d.line(pts, fill=int(rng.uniform(150, 255)), width=int(rng.uniform(3, 8) * S / 2))
    d.ellipse([cx - 62 * S, cy - 72 * S, cx + 62 * S, cy - 10 * S], fill=235)      # hair crown
    d.ellipse([cx - 40 * S, cy - 52 * S, cx + 40 * S, cy + 50 * S], fill=0)        # face (no ink)
    for ex in (-17, 17):
        d.ellipse([cx + (ex - 9) * S, cy - 12 * S, cx + (ex + 9) * S, cy - 3 * S], fill=255)  # eyes
        d.line([(cx + (ex - 11) * S, cy - 18 * S), (cx + (ex + 10) * S, cy - 20 * S)], fill=170, width=2 * S)  # brows
    d.line([(cx - 2 * S, cy - 4 * S), (cx - 5 * S, cy + 16 * S), (cx + 3 * S, cy + 17 * S)], fill=150, width=2 * S)
    d.ellipse([cx - 11 * S, cy + 26 * S, cx + 11 * S, cy + 33 * S], fill=210)       # lips
    a = np.asarray(im.resize((SW, SH), Image.LANCZOS)).astype(np.float32) / 255
    return gaussian_filter(a, .6)
FACE = face_target()
TEX = np.clip(.7 + .6 * gaussian_filter(rng.random((SH, SW)).astype(np.float32), 1.2) * 2 - .6, .5, 1.2)

def noise_psi(tsec):
    p = np.zeros((SH, SW), np.float32)
    for k, (kx, ky, w, a) in enumerate([(.07, .05, .21, 9), (.11, .13, .17, 6), (.19, .16, .29, 3.5), (.31, .27, .37, 2)]):
        p += a * np.sin(xx * kx + tsec * w + k) * np.cos(yy * ky - tsec * w * .7 + 2 * k)
    return p

def velocity(tsec):
    u = np.zeros((SH, SW), np.float32); v = np.zeros((SH, SW), np.float32)
    if tsec > T_HIT:
        for (t0, x0, sep, G, spd) in [(T_HIT, SW / 2, 7, 34, 1.0), (T_HIT + 3.5, SW / 2 - 4, 5, 18, .6)]:
            if tsec < t0: continue
            age = tsec - t0
            yc = 26 + spd * 95 * (1 - np.exp(-age / 6.0))
            sp = sep + age * .45
            g = G * np.exp(-age / 12)
            for sx, sg in ((-1, 1), (1, -1)):
                dx, dy = xx - (x0 + sx * sp), yy - yc
                r2 = dx * dx + dy * dy + 9
                u += -sg * g * dy / r2 * VSIGN; v += sg * g * dx / r2 * VSIGN
    psi = noise_psi(tsec) * (.08 + .02 * min(tsec, 20) / 20)
    u += np.gradient(psi, axis=0); v -= np.gradient(psi, axis=1)
    v += .16 + .1 * np.sin(xx * .21 + tsec * .4) * np.clip((tsec - 4) / 6, 0, 1)   # ink sinks in fingers
    return u, v

def step(tsec, dt=1.0):
    global D
    if tsec > T_FACE0 and tsec < T_FACE3:      # time slows while she forms
        dt = .25 if T_FACE1 < tsec < T_FACE2 else .6
    u, v = velocity(tsec)
    D = map_coordinates(D, [yy - v * dt, xx - u * dt], order=1, mode="constant").astype(np.float32)
    if T_HIT <= tsec < T_HIT + 1.2:
        D += 1.7 * np.exp(-(((xx - SW / 2) ** 2) / 10 + ((yy - 24) ** 2) / 14)) * (1 - (tsec - T_HIT) / 1.2)
    if tsec > T_FLOOD:                          # the water darkens
        D += .012 * (tsec - T_FLOOD) * (.4 + .6 * gaussian_filter(np.clip(D, 0, 1), 6))
    D = gaussian_filter(D, .12)
    D = np.clip(D, 0, 3)

# ------------------------------------------------------------------ render
RAMP = " .,:;i1tfLCG08@"
FONT = font(MONO, 22)
cap_f = font(MONOB, 44)
CAPS = [(.3, 4.0, "One drop of ink."),
        (4.5, 9.5, "Watch it closely."),
        (21.0, 24.0, "Did you see her?"),
        (24.2, T_BLACK, "She's been in the\nwater a long time.")]
YP, XP = np.mgrid[0:H, 0:W]
caust_base = rng.random((H // 30 + 2, W // 30 + 2))

def render(fi):
    tsec = fi / FPS
    step(tsec)
    Dv = np.clip(D, 0, 1)
    if T_FACE0 <= tsec < T_FACE3:
        a = np.interp(tsec, [T_FACE0, T_FACE1, T_FACE2, T_FACE3], [0, .9, .9, 0])
        Dv = np.clip(Dv * (1 - a * .75) + FACE * TEX * a, 0, 1)
    if T_FACE2 <= tsec < T_FACE2 + .1:          # imprint her into the ink as she dissolves
        D[:] = np.maximum(D, FACE * .9)
    cells = np.asarray(Image.fromarray((Dv * 255).astype(np.uint8)).resize((COLS, ROWS), Image.BOX)).astype(np.float32) / 255
    idx = np.clip((cells ** .8 * len(RAMP)).astype(int), 0, len(RAMP) - 1)
    chars = np.array(list(RAMP))[idx]
    # paper/water background with soft ink stain + moving caustics
    flood = np.clip((tsec - T_FLOOD) / (T_BLACK - T_FLOOD), 0, 1)
    bgc = np.array([226, 221, 205], np.float32) * (1 - flood * .92)
    big = np.asarray(Image.fromarray((Dv * 255).astype(np.uint8)).resize((GW, GH), Image.BILINEAR).filter(ImageFilter.GaussianBlur(10))).astype(np.float32) / 255
    arr = np.ones((H, W, 3), np.float32) * bgc
    ph = tsec * .6
    caus = 1 + .018 * np.sin(XP[::4, ::4] * .006 + ph) * np.sin(YP[::4, ::4] * .004 - ph * .7) + .012 * np.sin((XP[::4, ::4] + YP[::4, ::4]) * .009 + ph * 1.3)
    arr *= np.repeat(np.repeat(caus, 4, 0), 4, 1)[:H, :W, None]
    arr[GY:GY + GH, GX:GX + GW] *= (1 - .38 * big[..., None])
    m = ascii_mask(["".join(r) for r in chars], GX, GY, FONT, CH)
    face_k = np.interp(tsec, [T_FACE0, T_FACE1, T_FACE2, T_FACE3 + 2], [0, 1, 1, .4]) if tsec >= T_FACE0 else 0
    ink = np.array([20, 24, 40]) * (1 - face_k) + np.array([92, 10, 16]) * face_k
    if flood: ink = ink * (1 - flood) + np.array([60, 8, 12]) * flood
    arr = arr * (1 - m[..., None]) + ink * m[..., None]
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    # the falling drop
    if T_DROP <= tsec < T_HIT:
        k = (tsec - T_DROP) / (T_HIT - T_DROP)
        y = GY - 200 + k * (200 + 24 / SH * GH); x = W / 2
        d.ellipse([x - 9, y - 14, x + 9, y + 9], fill=(18, 20, 34))
        d.line([(x, y - 60), (x, y - 16)], fill=(70, 70, 80), width=2)
    if T_HIT <= tsec < T_HIT + 1.0:              # ripple rings at impact
        for k in range(3):
            rr = (tsec - T_HIT) * 420 - k * 60
            if rr > 0:
                al = max(0, 1 - (tsec - T_HIT)) * 120
                d.ellipse([W / 2 - rr, GY + 24 / SH * GH - rr * .18, W / 2 + rr, GY + 24 / SH * GH + rr * .18],
                          outline=(int(bgc[0] - al), int(bgc[1] - al), int(bgc[2] - al)), width=2)
    capcol = (30, 28, 34) if flood < .5 else (220, 200, 195)
    captions(d, tsec, CAPS, cap_f, color=capcol, y1=150, y2=180)
    return im

if os.environ.get('SIMONLY'):
    snaps = []
    for fi in range(int(15 * FPS)):
        step(fi / FPS)
        if fi % 60 == 30: snaps.append(Image.fromarray((255 - np.clip(D, 0, 1) * 255).astype(np.uint8)).resize((232, 348)))
    c = Image.new('L', (232 * len(snaps), 348)); [c.paste(im, (k * 232, 0)) for k, im in enumerate(snaps)]
    c.save('sim.png'); sys.exit()
checks = [.8, 1.6, 4, 9, 14, 16.2, 18, 19.8, 22, 26, T_BLACK + 1.0, 30]
wr = Writer(OUT, "ink_in_water.mp4", os.path.join(OUT, "ink.wav"), checks)
for fi in range(int(DUR * FPS)):
    tsec = fi / FPS
    if tsec >= T_BLACK:
        step(tsec)
        wr.write(end_card(tsec, T_BLACK + 1.4, bg=(4, 2, 3)))
    else:
        wr.write(render(fi))
wr.close()
print("done")
