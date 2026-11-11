# The Self-Writing Script

▶ [Watch on YouTube](https://youtube.com/shorts/1BmURdQPJpc)

A Python file types itself in a code editor at 3:13 AM. Its comments start talking to you, it runs, and then it deletes what it wrote.

**How it works**
- The editor, syntax highlighting, cursor and typing rhythm are all drawn with Pillow. There is no screen recording.
- Each keystroke has a randomized delay, and comments type at a different speed from code, so it feels like a person typing.
- Every keystroke triggers a synthesized key click, over a filtered-noise fan and a 50 Hz mains hum.

```bash
python shorts/selfwrite/selfwrite.py    # -> self_writing_script.mp4
```
