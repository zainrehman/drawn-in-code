# Garden of Life

▶ [Watch on YouTube](https://youtube.com/shorts/eS2qFFZcIv8)

Conway's Game of Life grows a garden of glowing flowers, which rot into a skull and go still. Then one glider escapes.

**How it works**
- A real Game of Life simulation, with slightly relaxed rules inside flower-shaped masks so the cells "want" to grow there.
- The masks change over time (stems, then petals, then a skull), so the same simulation tells the story.
- Every cell birth becomes a plucked note, so the garden composes its own music. The key slides to minor as it rots.

```bash
python shorts/garden/garden.py              # full render -> garden_of_life.mp4
SIMONLY=1 python shorts/garden/garden.py    # quick simulation preview -> sim.png
```
