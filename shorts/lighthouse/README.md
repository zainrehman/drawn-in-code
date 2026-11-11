# The Lighthouse

▶ [Watch on YouTube](https://youtube.com/shorts/KdAWNlCNSdo)

An ASCII night sea, looking back at a lighthouse on a cliff. Each time the beam swings round it lights the water, and the figures standing in it are closer. On the last turn the beam stops, pointed at you.

**How it works**
- The beam angle is a simple function of time, and its glow is computed per character cell.
- The figures are always dark, so you only see them against water lit by the beam. They're moved closer between flashes.
- Waves, wind, crashes and the foghorn are filtered noise and oscillators.

```bash
python shorts/lighthouse/lighthouse.py                 # full render -> the_lighthouse.mp4
AUDIOONLY=1 python shorts/lighthouse/lighthouse.py     # rebuild just the audio
```
