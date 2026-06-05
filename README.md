# Boids: Emergent Flocking

Three simple rules (separation, alignment, cohesion) produce emergent flocking in a swarm of boids.

Part of my portfolio of small, from-scratch visualisations of computer-science ideas. Built on numpy and matplotlib, so every moving part is visible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python boids_flocking.py                  # live animated window
python boids_flocking.py --save out.gif   # export a looping GIF
python boids_flocking.py --save out.mp4   # smaller file, best for the web (needs ffmpeg)
```
