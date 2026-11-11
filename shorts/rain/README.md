# Rain on Glass

▶ [Watch on YouTube](https://youtube.com/shorts/b5CMiMRa9lU)

An ASCII window on a rainy night, looking out at a street lamp. Someone is standing out there. Each lightning flash shows them a little closer, until they're at the glass and a whisper asks to be let in.

**How it works**
- The view outside (sky, trees, the lamp and its light cone) is drawn as greyscale shapes, averaged into character cells and mapped to a light-to-heavy ASCII ramp.
- The fogged glass is its own layer that slowly mists back up. Simulated drops run down it, pause, wander sideways, and clear a trail you can see through.
- The figure is a silhouette that gets bigger at each lightning strike, so it only moves while you're blinded by the flash.
- Rain hiss, drips, thunder and the drone are filtered noise and sine waves. The "let me in" whisper is espeak-ng, slowed, and panned from left to right before it ends up right in front of you.

This script is self-contained and doesn't use `common.py`.

```bash
python shorts/rain/rain.py              # full render -> rain_on_glass.mp4
PREVIEW=1 python shorts/rain/rain.py    # just the check frames -> check_*.png
```
