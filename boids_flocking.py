from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.viz import PALETTE, ACCENTS

# ---- tunables -------------------------------------------------------------
N_BOIDS = 90
SIZE = 480                # square canvas (px) — card sized
MAX_SPEED = 4.2
NEIGHBOR_R = 60.0
SEP_R = 22.0
W_SEP, W_ALI, W_COH = 1.6, 1.0, 0.9
W_FLEE = 3.2
SAVE_FRAMES = 150


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def limit(vecs, m):
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    scale = np.where(n > m, m / np.maximum(n, 1e-9), 1.0)
    return vecs * scale


def step(pos, vel, predator):
    """One flocking update (vectorized O(N^2))."""
    diff = pos[:, None, :] - pos[None, :, :]          # i - j
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    neigh = dist < NEIGHBOR_R
    counts = neigh.sum(1, keepdims=True)
    safe = np.maximum(counts, 1)

    # cohesion: toward mean neighbour position
    mean_pos = (neigh[..., None] * pos[None, :, :]).sum(1) / safe
    coh = np.where(counts > 0, mean_pos - pos, 0.0)

    # alignment: toward mean neighbour velocity
    mean_vel = (neigh[..., None] * vel[None, :, :]).sum(1) / safe
    ali = np.where(counts > 0, mean_vel - vel, 0.0)

    # separation: push away from very close boids
    close = dist < SEP_R
    sep = (close[..., None] * diff /
           (dist[..., None] ** 2 + 1e-6)).sum(1)

    acc = W_COH * coh + W_ALI * ali + W_SEP * sep * 40.0

    if predator is not None:
        away = pos - predator
        d = np.linalg.norm(away, axis=1, keepdims=True)
        flee = np.where(d < 120, away / (d ** 2 + 1e-6) * 4000, 0.0)
        acc += W_FLEE * flee

    vel = limit(vel + acc * 0.05, MAX_SPEED)
    pos = (pos + vel) % SIZE                            # wrap around edges
    return pos, vel


def draw(surf, pygame, pos, vel, colors, trail_surf):
    # fade the trail layer slightly, then redraw boids on top
    trail_surf.fill((*hex_to_rgb(PALETTE["bg"]), 26))
    surf.blit(trail_surf, (0, 0))
    for p, v, c in zip(pos, vel, colors):
        ang = np.arctan2(v[1], v[0])
        tip = p + 7 * np.array([np.cos(ang), np.sin(ang)])
        left = p + 5 * np.array([np.cos(ang + 2.5), np.sin(ang + 2.5)])
        right = p + 5 * np.array([np.cos(ang - 2.5), np.sin(ang - 2.5)])
        pygame.draw.polygon(surf, c, [tip, left, right])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="OUT.gif")
    args = ap.parse_args()

    if args.save:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame
    pygame.init()
    if args.save:
        screen = pygame.Surface((SIZE, SIZE))
    else:
        screen = pygame.display.set_mode((SIZE, SIZE))
        pygame.display.set_caption("boids — emergent flocking")
    clock = pygame.time.Clock()

    trail = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)

    pos = np.random.rand(N_BOIDS, 2) * SIZE
    vel = (np.random.rand(N_BOIDS, 2) - 0.5) * MAX_SPEED
    colors = [hex_to_rgb(ACCENTS[i % len(ACCENTS)]) for i in range(N_BOIDS)]
    screen.fill(hex_to_rgb(PALETTE["bg"]))

    if args.save:
        import imageio.v2 as imageio
        frames = []
        for _ in range(SAVE_FRAMES):
            pos, vel = step(pos, vel, None)
            draw(screen, pygame, pos, vel, colors, trail)
            frames.append(np.transpose(pygame.surfarray.array3d(screen),
                                       (1, 0, 2)).copy())
        if args.save.lower().endswith(".gif"):
            imageio.mimsave(args.save, frames, fps=30, loop=0)
        else:                       # mp4 / webm via the ffmpeg writer
            imageio.mimsave(args.save, frames, fps=30)
        print(f"saved -> {args.save}")
        pygame.quit()
        return

    running = True
    while running:
        predator = None
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        if pygame.mouse.get_focused() and any(pygame.mouse.get_pressed()):
            predator = np.array(pygame.mouse.get_pos(), float)
        pos, vel = step(pos, vel, predator)
        draw(screen, pygame, pos, vel, colors, trail)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
