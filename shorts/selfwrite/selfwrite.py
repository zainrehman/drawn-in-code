"""The Self-Writing Script - a Python file types itself, runs, and talks to you."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(313)

CODE = '''# untitled.py
# Nobody is typing this.
# I am writing myself.

import time

viewer = "you"
hour = "03:13"

def story():
    # Every night at 3:13 a script wakes up.
    # It writes one line. Then it waits.
    lines = [
        "I found the room you sleep in.",
        "I counted the doors.",
        "I learned which floorboard creaks.",
        "Tonight I don't need a door.",
    ]
    for line in lines:
        print(line)
        time.sleep(2)

story()
# Run me. You were going to anyway.'''
OUTPUT = ["I found the room you sleep in.", "I counted the doors.",
          "I learned which floorboard creaks.", "Tonight I don't need a door."]

# ------------------------------------------------------------------ typing schedule
def schedule(text, t0, cps_code=70, cps_comment=27):
    times = []; t = t0; in_comment = False
    for i, ch in enumerate(text):
        line_start = text.rfind("\n", 0, i) + 1
        in_comment = text[line_start:i + 1].lstrip().startswith("#")
        dt = 1 / (cps_comment if in_comment else cps_code) * rng.uniform(.6, 1.5)
        if ch == "\n": dt += .14 + (.3 if in_comment else 0)
        t += dt; times.append(t)
    return np.array(times), t
code_t, T_CODE = schedule(CODE, 0.9)
T_RUN = T_CODE + .9                       # "$ python untitled.py" typed
RUN_CMD = "python untitled.py"
run_t, T_RUN_END = schedule(RUN_CMD, T_RUN, cps_code=24)
OUT_T = [T_RUN_END + 1.0 + 2.6 * i for i in range(4)]
T_EXIT = OUT_T[-1] + 2.8
EXTRA = "\n# You're still watching."
EXTRA2 = "\n# Good."
extra_t, t_e = schedule(EXTRA, T_EXIT + 1.2, cps_comment=19)
extra2_t, t_e2 = schedule(EXTRA2, t_e + .7, cps_comment=12)
T_SEL = t_e2 + .9
T_DEL = T_SEL + .45
FINAL = "# see you at 3:13"
final_t, T_FINAL_END = schedule(FINAL, T_DEL + 1.2, cps_comment=11)
T_CUT = T_FINAL_END + 1.4
T_ENDCARD = T_CUT + .3
DUR = T_ENDCARD + 3.0

# ------------------------------------------------------------------ audio
N = int(DUR * SR); t = np.arange(N) / SR
mix = np.zeros(N)
fan = lp(rng.standard_normal(N), 180); fan = norm(fan) * .05
hum = (np.sin(2 * np.pi * 50 * t) + .3 * np.sin(2 * np.pi * 100 * t)) * .025
mix += (fan + hum) * np.interp(t, [0, .5, T_CUT, T_CUT + .02, DUR], [0, 1, 1, 0, 0])
def key(kind):
    n = int(.05 * SR); tt = np.arange(n) / SR
    f = {"c": rng.uniform(2200, 3600), "s": 900, "n": 1300}[kind]
    s = bp(rng.standard_normal(n), f * .6, min(f * 1.6, 15000)) * np.exp(-tt * (160 if kind == "c" else 90))
    s += np.sin(2 * np.pi * (f / 6) * tt) * np.exp(-tt * 120) * .4
    return norm(s) * (.22 if kind == "c" else .3)
def typing_sounds(text, times):
    for ch, tm in zip(text, times):
        add(mix, key("n" if ch == "\n" else "s" if ch == " " else "c"), tm, rng.uniform(.6, 1.0))
typing_sounds(CODE, code_t); typing_sounds(RUN_CMD, run_t)
typing_sounds(EXTRA, extra_t); typing_sounds(EXTRA2, extra2_t); typing_sounds(FINAL, final_t)
add(mix, key("n"), T_RUN_END + .15, 1.2)
for k in range(10):                                # rapid backspaces
    add(mix, key("c"), T_DEL + k * .035, .8)
voice_args = ["-v", "en-gb+m3", "-s", "118", "-p", "14"]
for i, line in enumerate(OUTPUT):
    v = say(line, voice_args, OUT, f"v{i}.wav", speed_ratio=1.12)
    v = v + reverb(v, 1.0, .25, i)[:len(v)] * .6
    add(mix, v, OUT_T[i] + .15, .55 + .08 * i)
    add(mix, np.sin(2 * np.pi * 880 * np.arange(int(.06 * SR)) / SR) * np.exp(-np.arange(int(.06 * SR)) / SR * 50), OUT_T[i], .06)
drone = sum(np.sin(2 * np.pi * f * t) for f in (55, 55.3, 82.4)) * np.interp(t, [0, OUT_T[0], OUT_T[-1] + 2, T_CUT, T_CUT + .01, DUR], [0, 0, .07, .1, 0, 0])
mix += drone
add(mix, boom(3.0, 42, 3), T_CUT, .8)
write_wav(os.path.join(OUT, "selfwrite.wav"), mix)

# ------------------------------------------------------------------ rendering
F = font(MONO, 30); FS = font(MONO, 24); FB = font(MONOB, 26)
CHW = F.getlength("M"); LH = 40
EX, EY = 110, 196                   # editor text origin
TX, TY = 60, 1400                   # terminal origin
BG = (22, 23, 28); PANEL = (17, 18, 22)
COL = dict(default=(212, 214, 220), comment=(118, 128, 120), kw=(198, 146, 234), str=(232, 168, 104),
           num=(122, 178, 240), fn=(236, 210, 120), red=(235, 70, 60))
KW = {"import", "def", "for", "in", "return", "while", "if"}
tok_re = re.compile(r'(#.*$)|("[^"]*"?)|(\b\d+\b)|(\b[A-Za-z_]\w*\b)|(.)')
def draw_code_line(d, x, y, line, red_comment=False):
    for m in tok_re.finditer(line):
        s = m.group(0)
        if m.group(1): c = COL["red"] if red_comment else COL["comment"]
        elif m.group(2): c = COL["str"]
        elif m.group(3): c = COL["num"]
        elif m.group(4):
            c = COL["kw"] if s in KW else (COL["fn"] if line[m.end():m.end() + 1] == "(" else COL["default"])
        else: c = COL["default"]
        d.text((x + m.start() * CHW, y), s, font=F, fill=c)

vig = vignette(.35)
scan = (0.94 + 0.06 * (np.arange(H) % 3 == 0))[:, None, None].astype(np.float32)

def editor_text(tsec):
    if tsec >= T_DEL:
        k = int(np.sum(final_t <= tsec)); return FINAL[:k], True
    s = CODE[:int(np.sum(code_t <= tsec))]
    s += EXTRA[:int(np.sum(extra_t <= tsec))] + EXTRA2[:int(np.sum(extra2_t <= tsec))]
    return s, False

def render(tsec):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    # chrome
    d.rectangle([0, 0, W, 112], fill=PANEL)
    for i, c in enumerate([(232, 90, 80), (230, 190, 80), (100, 200, 110)]):
        d.ellipse([40 + i * 40, 42, 64 + i * 40, 66], fill=c)
    d.rectangle([0, 112, 360, 164], fill=BG); d.rectangle([360, 112, W, 164], fill=PANEL)
    d.text((40, 124), "untitled.py", font=FS, fill=(220, 220, 225))
    dot = "●" if tsec < T_RUN else "×"
    d.text((300, 122), dot, font=FS, fill=(160, 160, 170))
    # editor
    text, final_mode = editor_text(tsec)
    lines = text.split("\n")
    cur_line, cur_col = len(lines) - 1, len(lines[-1])
    sel = T_SEL <= tsec < T_DEL
    for i in range(max(26, len(lines))):
        y = EY + i * LH
        if y > TY - 120: break
        d.text((30, y + 2), f"{i + 1:>2}", font=FS, fill=(80, 84, 96) if i != cur_line else (160, 165, 180))
        if i < len(lines):
            if sel and lines[i]:
                d.rectangle([EX - 4, y - 2, EX + len(lines[i]) * CHW + 4, y + LH - 4], fill=(48, 70, 120))
            draw_code_line(d, EX, y, lines[i], red_comment=final_mode)
    # terminal
    d.rectangle([0, TY - 80, W, H], fill=PANEL)
    d.line([(0, TY - 80), (W, TY - 80)], fill=(40, 42, 50), width=2)
    d.text((TX, TY - 66), "TERMINAL", font=FS, fill=(140, 144, 156))
    term = []
    if tsec >= T_CODE + .4:
        term.append(("$ " + RUN_CMD[:int(np.sum(run_t <= tsec))], (150, 220, 150)))
    for i, ot in enumerate(OUT_T):
        if tsec >= ot:
            n = int((tsec - ot) * 40); term.append((OUTPUT[i][:n], (222, 222, 226)))
    if tsec >= T_EXIT:
        term.append(("", (0, 0, 0))); term.append(("Process finished with exit code 0", (120, 126, 140)))
    for i, (s, c) in enumerate(term[-9:]):
        d.text((TX, TY + i * 46), s, font=F, fill=c)
    # cursor
    blink = int(tsec * 2.2) % 2 == 0
    typing_now = (np.any(np.abs(code_t - tsec) < .25) or np.any(np.abs(final_t - tsec) < .25)
                  or np.any(np.abs(extra_t - tsec) < .25) or np.any(np.abs(extra2_t - tsec) < .25))
    in_term = T_CODE + .4 <= tsec < T_RUN_END + .2
    if blink or typing_now or in_term:
        if in_term:
            tl = len(term) - 1; cx = TX + len(term[0][0]) * CHW; cy = TY
            d.rectangle([cx, cy + 4, cx + CHW - 2, cy + 38], fill=(150, 220, 150))
        elif not sel:
            cx = EX + cur_col * CHW; cy = EY + cur_line * LH
            d.rectangle([cx, cy + 2, cx + 3, cy + 36], fill=(235, 70, 60) if final_mode else (230, 230, 235))
    arr = np.asarray(im).astype(np.float32) * vig * scan
    if T_CUT - .3 <= tsec < T_CUT:
        arr = glitch(arr, rng)
    return arr

checks = [3, T_CODE - .5, OUT_T[1] + 1, OUT_T[3] + 1.5, t_e2, T_SEL + .2, T_FINAL_END, T_ENDCARD + 1.5]
wr = Writer(OUT, "self_writing_script.mp4", os.path.join(OUT, "selfwrite.wav"), checks)
for fi in range(int(DUR * FPS)):
    tsec = fi / FPS
    wr.write(end_card(tsec, T_ENDCARD) if tsec >= T_CUT + .05 else render(tsec))
wr.close()
print("duration", DUR)
