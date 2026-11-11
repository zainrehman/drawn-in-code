# Ink in Water

▶ [Watch on YouTube](https://youtube.com/shorts/7_bCSzTTlQg)

A single drop of glyph-ink falls into pale water and blooms. For a few seconds the ink stops being ink and forms a face, then smears back into the swirl.

**How it works**
- The ink is a density field moved by a vortex ring plus curl noise, using semi-Lagrangian advection.
- The density is drawn as ASCII characters, from light to heavy glyphs.
- The face is a drawn target shape that is blended into the ink density for a few seconds, while the simulation slows down, then released back into the flow.
- The audio (the plop, bubbles, glassy pentatonic tones, heartbeat and choir) is all synthesized.

```bash
python shorts/ink/ink.py              # full render -> ink_in_water.mp4
SIMONLY=1 python shorts/ink/ink.py    # quick simulation preview -> sim.png
```
