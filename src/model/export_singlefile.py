"""
Export a trained TinyQuoridorNet checkpoint into a self-contained, single-file Python agent
that runs in < 1s on CPU using only the Python standard library (no numpy, no torch, no jax).
Patterned similarly to delta.py: int8 quantized weights compressed with zlib + base64.

Usage:
  python -m model.export_singlefile runs/tiny_run/latest -o agent.py
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import zlib

import numpy as np
from flax import nnx

from model.checkpoint import load_model


TEMPLATE = '''# Self-contained Quoridor Agent (Standard Library Only)
# Generated from QuoriZero TinyQuoridorNet weights
import base64
import math
import zlib
from dataclasses import dataclass, replace

BOARD_SIZE = 9
WALL_SLOTS = 8
WALLS_PER_PLAYER = 10
STARTS = ((0, 4), (8, 4))
GOAL_ROWS = (8, 0)
PAWN_OFFSETS = (
    (-1, 0), (1, 0), (0, 1), (0, -1),
    (-2, 0), (2, 0), (0, 2), (0, -2),
    (-1, 1), (-1, -1), (1, 1), (1, -1),
)
N_PAWN = 12
N_WALL = 64
ACTION_SIZE = 140

# --- WEIGHT BLOBS ---
WEIGHTS_META = {meta_json}
WEIGHTS_BLOB = "{weights_blob}"


def _decompress_weights():
    raw = zlib.decompress(base64.b64decode(WEIGHTS_BLOB))
    offset = 0
    weights = {{}}
    for name, shape, scale in WEIGHTS_META:
        size = 1
        for s in shape:
            size *= s
        chunk = raw[offset : offset + size]
        offset += size
        signed = [((b ^ 128) - 128) * scale for b in chunk]
        if len(shape) == 1:
            weights[name] = signed
        elif len(shape) == 2:
            r, c = shape
            weights[name] = [signed[i * c : (i + 1) * c] for i in range(r)]
    return weights


NET = _decompress_weights()


def _dense(x, W, b):
    # W is (in_dim, out_dim), x is (in_dim,), b is (out_dim,)
    out_dim = len(b)
    in_dim = len(x)
    out = [0.0] * out_dim
    for i in range(in_dim):
        xi = x[i]
        if xi != 0.0:
            row = W[i]
            for j in range(out_dim):
                out[j] += xi * row[j]
    for j in range(out_dim):
        out[j] += b[j]
    return out


def _relu(x):
    return [v if v > 0.0 else 0.0 for v in x]


def forward_net(planes):
    # planes: flat list of 729 floats
    h = _dense(planes, NET["fc1_w"], NET["fc1_b"])
    h = _relu(h)
    h = _dense(h, NET["fc2_w"], NET["fc2_b"])
    h = _relu(h)
    logits = _dense(h, NET["p_w"], NET["p_b"])
    vm = _dense(h, NET["vm_w"], NET["vm_b"])
    value = math.tanh(vm[0])
    return logits, value


# --- QUORIDOR ENGINE (STDLIB) ---
@dataclass(frozen=True)
class GameState:
    pawns: tuple[tuple[int, int], tuple[int, int]]
    walls: frozenset[tuple[str, int, int]]
    walls_left: tuple[int, int]
    to_move: int


def shortest_distance(walls, start, goal_row):
    visited = {{start}}
    queue = [(start[0], start[1], 0)]
    while queue:
        r, c, d = queue.pop(0)
        if r == goal_row:
            return d
        # Explore 4 directions
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                # check wall block
                if dr == -1 and (("H", r - 1, c) in walls or ("H", r - 1, c - 1) in walls):
                    continue
                if dr == 1 and (("H", r, c) in walls or ("H", r, c - 1) in walls):
                    continue
                if dc == -1 and (("V", r, c - 1) in walls or ("V", r - 1, c - 1) in walls):
                    continue
                if dc == 1 and (("V", r, c) in walls or ("V", r - 1, c) in walls):
                    continue
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, d + 1))
    return None


def open_steps(walls, r, c):
    res = []
    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 9 and 0 <= nc < 9:
            if dr == -1 and (("H", r - 1, c) in walls or ("H", r - 1, c - 1) in walls):
                continue
            if dr == 1 and (("H", r, c) in walls or ("H", r, c - 1) in walls):
                continue
            if dc == -1 and (("V", r, c - 1) in walls or ("V", r - 1, c - 1) in walls):
                continue
            if dc == 1 and (("V", r, c) in walls or ("V", r - 1, c) in walls):
                continue
            res.append((nr, nc))
    return res


def pawn_moves(state):
    me, opp = state.pawns[state.to_move], state.pawns[1 - state.to_move]
    r, c = me
    valid = []
    for step in open_steps(state.walls, r, c):
        if step != opp:
            valid.append(step)
        else:
            dr, dc = opp[0] - r, opp[1] - c
            jump = (opp[0] + dr, opp[1] + dc)
            if jump in open_steps(state.walls, opp[0], opp[1]):
                valid.append(jump)
            else:
                for diag in open_steps(state.walls, opp[0], opp[1]):
                    if diag != me:
                        valid.append(diag)
    return valid


def wall_placements(state):
    if state.walls_left[state.to_move] <= 0:
        return []
    blocked = set()
    for o, r, c in state.walls:
        blocked.add((o, r, c))
        blocked.add(("V" if o == "H" else "H", r, c))
        if o == "H":
            blocked.add(("H", r, c - 1))
            blocked.add(("H", r, c + 1))
        else:
            blocked.add(("V", r - 1, c))
            blocked.add(("V", r + 1, c))

    valid = []
    for o in ("H", "V"):
        for r in range(8):
            for c in range(8):
                w = (o, r, c)
                if w in blocked:
                    continue
                new_walls = state.walls | {{w}}
                if shortest_distance(new_walls, state.pawns[0], GOAL_ROWS[0]) is not None and \\
                   shortest_distance(new_walls, state.pawns[1], GOAL_ROWS[1]) is not None:
                    valid.append(w)
    return valid


def encode_action(state, action_type, detail):
    if action_type == "move":
        frm = state.pawns[state.to_move]
        offset = (detail[0] - frm[0], detail[1] - frm[1])
        return PAWN_OFFSETS.index(offset)
    orient, r, c = detail
    base = N_PAWN if orient == "H" else N_PAWN + N_WALL
    return base + r * 8 + c


def decode_action(state, action):
    if action < N_PAWN:
        dr, dc = PAWN_OFFSETS[action]
        r, c = state.pawns[state.to_move]
        return "move", (r + dr, c + dc)
    idx = action - N_PAWN
    orient = "H" if idx < N_WALL else "V"
    idx %= N_WALL
    return "wall", (orient, idx // 8, idx % 8)


def step(state, action):
    atype, val = decode_action(state, action)
    if atype == "move":
        pawns = list(state.pawns)
        pawns[state.to_move] = val
        next_s = replace(state, pawns=(pawns[0], pawns[1]), to_move=1 - state.to_move)
        done = val[0] == GOAL_ROWS[state.to_move]
        return next_s, done
    left = list(state.walls_left)
    left[state.to_move] -= 1
    next_s = replace(
        state,
        walls=state.walls | {{val}},
        walls_left=(left[0], left[1]),
        to_move=1 - state.to_move,
    )
    return next_s, False


def extract_features(state):
    # Produce flat 729 input array (9 planes of 9x9)
    # Planes:
    # 0: mover pawn, 1: opp pawn, 2: h_walls, 3: v_walls,
    # 4: mover walls_left, 5: opp walls_left, 6: mover dist, 7: opp dist, 8: constant 1
    # Canonical: if mover is player 0, flipped to player's perspective
    p0, p1 = state.pawns
    w0, w1 = state.walls_left
    walls = state.walls
    flip = state.to_move == 0

    def flip_r(r):
        return 8 - r if flip else r

    # pawns
    my_r, my_c = flip_r(state.pawns[state.to_move][0]), state.pawns[state.to_move][1]
    op_r, op_c = flip_r(state.pawns[1 - state.to_move][0]), state.pawns[1 - state.to_move][1]

    # distances
    dist0 = [[shortest_distance(walls, (r, c), GOAL_ROWS[state.to_move]) or 32 for c in range(9)] for r in range(9)]
    dist1 = [[shortest_distance(walls, (r, c), GOAL_ROWS[1 - state.to_move]) or 32 for c in range(9)] for r in range(9)]

    arr = []
    for r in range(9):
        for c in range(9):
            rr = 8 - r if flip else r
            # Plane 0: my pawn
            p0_val = 1.0 if (r, c) == (my_r, my_c) else 0.0
            # Plane 1: opp pawn
            p1_val = 1.0 if (r, c) == (op_r, op_c) else 0.0
            # Plane 2: h wall
            hw = 1.0 if (("H", 7 - rr if flip else rr, c) in walls) else 0.0
            # Plane 3: v wall
            vw = 1.0 if (("V", 7 - rr if flip else rr, c) in walls) else 0.0
            # Plane 4, 5: walls left
            wl0 = state.walls_left[state.to_move] / 10.0
            wl1 = state.walls_left[1 - state.to_move] / 10.0
            # Plane 6, 7: dist
            d0 = min(dist0[rr][c], 32) / 16.0
            d1 = min(dist1[rr][c], 32) / 16.0
            # Plane 8: const 1
            arr.extend([p0_val, p1_val, hw, vw, wl0, wl1, d0, d1, 1.0])
    return arr


def choose_move(state):
    # Legal actions
    moves = [encode_action(state, "move", m) for m in pawn_moves(state)]
    walls = [encode_action(state, "wall", w) for w in wall_placements(state)]
    legal = moves + walls
    if not legal:
        return None

    # Immediate win check
    for a in moves:
        _, done = step(state, a)
        if done:
            return a

    # Neural network policy prior
    features = extract_features(state)
    logits, val = forward_net(features)

    # Pick highest logit among legal actions
    best_a = legal[0]
    best_score = -float("inf")
    for a in legal:
        score = logits[a]
        if score > best_score:
            best_score = score
            best_a = a
    return best_a
'''


def quantize_and_pack(model):
    state = nnx.state(model)
    meta = []
    blobs = []

    def process(name, arr):
        nonlocal blobs
        arr = np.asarray(arr, dtype=np.float32)
        scale = float(np.max(np.abs(arr)) / 127.0) if np.max(np.abs(arr)) > 0 else 1.0
        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
        # Shift to uint8 for byte packing
        u = (q.astype(np.int16) + 128) ^ 128
        blobs.append(bytes(u.astype(np.uint8).flatten()))
        meta.append((name, list(arr.shape), scale))

    process("fc1_w", state.fc1.kernel.value)
    process("fc1_b", state.fc1.bias.value)
    process("fc2_w", state.fc2.kernel.value)
    process("fc2_b", state.fc2.bias.value)
    process("p_w", state.policy_out.kernel.value)
    process("p_b", state.policy_out.bias.value)
    process("vm_w", state.val_margin_out.kernel.value)
    process("vm_b", state.val_margin_out.bias.value)

    packed = b"".join(blobs)
    compressed = zlib.compress(packed, level=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    return meta, b64


def main():
    p = argparse.ArgumentParser(description="Export TinyQuoridorNet checkpoint to single-file agent")
    p.add_argument("checkpoint", help="Path to checkpoint directory (e.g. runs/tiny/latest)")
    p.add_argument("-o", "--output", default="agent.py", help="Output .py file path")
    args = p.parse_args()

    model = load_model(args.checkpoint)
    meta, blob = quantize_and_pack(model)

    code = TEMPLATE.format(
        meta_json=json.dumps(meta),
        weights_blob=blob,
    )
    out_path = Path(args.output)
    out_path.write_text(code, encoding="utf-8")
    print(f"Exported self-contained agent to {out_path} ({len(code)} bytes)")


if __name__ == "__main__":
    main()
