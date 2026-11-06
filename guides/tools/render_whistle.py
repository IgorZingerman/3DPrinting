#!/usr/bin/env python3
"""Draw the pictures for the whistle guide from the guide's own measurements.

Every shape below uses the same sizes and positions as the steps in
whistle-tinkercad.html, so a picture can't drift away from the instructions.
Change a size in the guide? Change it here too and run this again.

    pip install numpy
    python3 guides/tools/render_whistle.py

Writes PNGs and anchors.json (where to put labels) to guides/images/whistle/.
These are computer drawings, not TinkerCad screenshots.

Directions match the guide: x = left/right on screen (mouthpiece on the left),
y = up/down on screen when looking from the top (window side is "back", +y),
z = height. The barrel is centered on x=0, y=0.
"""
import json
import pathlib
import struct
import zlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "images" / "whistle"
SS = 2  # draw at twice the size, then shrink, for smooth edges


def hx(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255 for i in (0, 2, 4)])


ORANGE, TAN = hx("f4a259"), hx("eccfa8")
BLUE, PALEBLUE = hx("2a6fdb"), hx("8fb4ea")
INK = hx("1f2a44")
PLANE, GRID = hx("e4edf8"), hx("c3d3e8")


# ---- shapes: each returns the distance from points P to its surface ----
def cyl(cx, cy, r, z0, z1):
    def f(P):
        a = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
        b = np.maximum(z0 - P[:, 2], P[:, 2] - z1)
        return np.minimum(np.maximum(a, b), 0) + np.hypot(np.maximum(a, 0), np.maximum(b, 0))
    return f


def box(x0, x1, y0, y1, z0, z1):
    c = np.array([x0 + x1, y0 + y1, z0 + z1]) / 2
    h = np.array([x1 - x0, y1 - y0, z1 - z0]) / 2

    def f(P):
        q = np.abs(P - c) - h
        return np.minimum(q.max(axis=1), 0) + np.linalg.norm(np.maximum(q, 0), axis=1)
    return f


def union(*fs):
    return lambda P: np.min([f(P) for f in fs], axis=0)


def minus(a, *bs):
    u = union(*bs)
    return lambda P: np.maximum(a(P), -u(P))


# ---- half a whistle, exactly as the steps build it ----
BARREL = cyl(0, 0, 13, 0, 10)                 # 26 x 26 x 10
MOUTH = box(-26, 0, 7, 13, 0, 10)             # 26 x 6 x 10, aligned right+top, then left 13
RING = minus(cyl(15, 0, 4, 0, 10), cyl(15, 0, 2, -1, 12))  # 8 x 8 x 10 with a 4 x 4 x 12 hole, right 15
CHAMBER = cyl(0, 0, 11, 2, 14)                # 22 x 22 x 12, lifted 2
TUNNEL = box(-28, 0, 10, 12, 2, 14)           # 28 x 2 x 12, aligned right+top, down 1, lifted 2
WINDOW = cyl(-2, 15, 4.5, 2, 14)              # 9 x 9 x 12, aligned middle+middle to the barrel, left 2, up 15, lifted 2
                                              # Its curve leaves a thin, sharp tip at about (0, 11), mid air stream.
HALF = minus(union(BARREL, MOUTH, RING), CHAMBER, TUNNEL, WINDOW)


def flipped(f, height):
    """The half flipped upside down, with its bottom lifted to this height."""
    def g(P):
        Q = P.copy()
        Q[:, 2] = 10 + height - P[:, 2]
        return f(Q)
    return g


TOP = flipped(HALF, 10)
WHISTLE = union(HALF, TOP)

BOUNDS = [(-28, -13, 0), (19, 14, 20)]


# ---- camera ----
def camera(az, el, pad=0.08, width=800, height=520, center=None, span=None):
    """span = how many mm wide the picture is (leave out to fit the whole whistle)."""
    a, e = np.radians(az), np.radians(el)
    to_cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross([0, 0, 1], to_cam)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(to_cam, right)
    lo, hi = BOUNDS
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u, v = corners @ right, corners @ up
    cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    scale = max((u.max() - u.min()) / width, (v.max() - v.min()) / height) * (1 + 2 * pad)
    if span is not None:
        scale = span / width
    if center is not None:
        cu, cv = np.dot(center, right), np.dot(center, up)
    return dict(right=right, up=up, to_cam=to_cam, cu=cu, cv=cv, scale=scale, w=width, h=height)


def project(cam, pt):
    u, v = np.dot(pt, cam["right"]), np.dot(pt, cam["up"])
    return [round(float((u - cam["cu"]) / cam["scale"] + cam["w"] / 2), 1),
            round(float(cam["h"] / 2 - (v - cam["cv"]) / cam["scale"]), 1)]


# ---- drawing ----
def trace(f, ro, rd, tmax=420.0):
    t, hit = np.zeros(len(ro)), np.zeros(len(ro), bool)
    act = np.arange(len(ro))
    for _ in range(110):
        if not len(act):
            break
        d = f(ro[act] + rd[act] * t[act, None])
        hit[act[d < 0.004]] = True
        t[act] += np.maximum(d, 0) * 0.98
        act = act[(d >= 0.004) & (t[act] < tmax)]
    t[~hit] = np.inf
    return t


def normals(f, P, e=0.012):
    n = np.stack([f(P + d) - f(P - d) for d in np.eye(3) * e], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def edges(ids, depth, nrm, depth_gap=0.7):
    """Find outlines: where the shape, depth or surface direction jumps."""
    e = np.zeros(ids.shape, bool)
    for ax in (0, 1):
        di = np.diff(ids, axis=ax) != 0
        with np.errstate(invalid="ignore"):
            dd = np.abs(np.diff(depth, axis=ax)) > depth_gap
        dot = (np.take(nrm, range(1, nrm.shape[ax]), axis=ax) * np.take(nrm, range(nrm.shape[ax] - 1), axis=ax)).sum(-1)
        jump = di | dd | (dot < 0.75)
        pad = [(0, 0), (0, 0)]
        pad[ax] = (0, 1)
        e |= np.pad(jump, pad)
    return e


def draw(name, cam, solids=(), holes=()):
    """solids and holes are lists of (shape, color)."""
    W, H = cam["w"] * SS, cam["h"] * SS
    xs = (np.arange(W) + 0.5) / SS - cam["w"] / 2
    ys = cam["h"] / 2 - (np.arange(H) + 0.5) / SS
    U, V = np.meshgrid(xs * cam["scale"] + cam["cu"], ys * cam["scale"] + cam["cv"])
    rd0 = -cam["to_cam"]
    ro = (U.ravel()[:, None] * cam["right"] + V.ravel()[:, None] * cam["up"]
          + cam["to_cam"] * 200)
    rd = np.tile(rd0, (len(ro), 1))
    light = np.array([-0.35, -0.55, 0.76])
    light = light / np.linalg.norm(light)
    fill = np.array([0.7, 0.5, 0.3])
    fill = fill / np.linalg.norm(fill)

    N = len(ro)
    img = np.zeros((N, 3))
    tp = np.full(N, np.inf)
    if rd0[2] < 0:
        tp[:] = -ro[:, 2] / rd0[2]
    Pp = ro + rd * np.where(np.isfinite(tp), tp, 0)[:, None]
    gx = np.abs((Pp[:, 0] + 2.5) / 5 - np.round((Pp[:, 0] + 2.5) / 5)) * 5
    gy = np.abs((Pp[:, 1] + 2.5) / 5 - np.round((Pp[:, 1] + 2.5) / 5)) * 5
    img[:] = PLANE
    img[(gx < 0.07) | (gy < 0.07)] = GRID

    ids, depth, nrm = np.zeros(N, int), tp.copy(), np.zeros((N, 3))
    nrm[:, 2] = 1
    best = np.full(N, np.inf)
    sidx = np.full(N, -1)
    for i, (f, _) in enumerate(solids):
        t = trace(f, ro, rd)
        m = t < best
        best[m], sidx[m] = t[m], i
    m = (best < tp)
    allsolid = union(*[f for f, _ in solids]) if solids else None
    if m.any():
        P = ro[m] + rd[m] * best[m, None]
        n = normals(allsolid, P)
        col = np.array([solids[i][1] for i in sidx[m]])
        lit = (0.52 + 0.30 * np.maximum(n @ light, 0) + 0.12 * np.maximum(n @ fill, 0)
               + 0.22 * np.maximum(n @ cam["to_cam"], 0))
        sh = trace(allsolid, P + n * 0.03, np.tile(light, (len(P), 1)), 120)
        lit = lit - 0.12 * np.isfinite(sh)
        img[m] = col * lit[:, None]
        ids[m], depth[m], nrm[m] = sidx[m] + 1, best[m], n
    # ground shadow
    if solids:
        g = ~m & np.isfinite(tp)
        sh = trace(allsolid, Pp[g] + np.array([0, 0, 0.02]), np.tile(light, (g.sum(), 1)), 120)
        idx = np.where(g)[0][np.isfinite(sh)]
        img[idx] *= 0.9

    def to_img(a):
        return a.reshape(H, W, *a.shape[1:])

    I = to_img(img)
    e = edges(to_img(ids), to_img(depth), to_img(nrm))
    I[e] = I[e] * 0.1 + INK * 0.9

    # holes: see-through blue, so kids can spot them inside the solid parts
    for f, col in holes:
        t = trace(f, ro, rd)
        hm = np.isfinite(t)
        if not hm.any():
            continue
        P = ro[hm] + rd[hm] * t[hm, None]
        n = normals(f, P)
        lit = 0.74 + 0.30 * np.maximum(n @ light, 0)
        hc = np.zeros((N, 3))
        hc[hm] = np.clip(col * lit[:, None], 0, 1)
        front = t < best
        alpha = np.where(hm, np.where(front, 0.8, 0.62), 0)
        A = to_img(alpha)[..., None]
        I = I * (1 - A) + to_img(hc) * A
        hn = np.zeros((N, 3))
        hn[:, 2] = 1
        hn[hm] = n
        ht = t.copy()
        he = edges(to_img(hm.astype(int)), to_img(np.where(hm, t, 0)), to_img(hn), 0.7)
        I[he] = I[he] * 0.15 + hx("174a9c") * 0.85

    save(name, I, cam["w"], cam["h"])


def save(name, I, w, h):
    """Shrink to final size and write a PNG."""
    small = I.reshape(h, SS, w, SS, 3).mean(axis=(1, 3))
    rgb = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
    raw = b"".join(b"\0" + row.tobytes() for row in rgb)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    (OUT / f"{name}.png").write_bytes(png)
    print("wrote", name)


def slice_view(name, f, z, x0, x1, y0, y1, px_per_mm):
    """A flat slice through the model at height z, seen from the top."""
    w, h = round((x1 - x0) * px_per_mm), round((y1 - y0) * px_per_mm)
    xs = x0 + (np.arange(w * SS) + 0.5) / (SS * px_per_mm)
    ys = y1 - (np.arange(h * SS) + 0.5) / (SS * px_per_mm)
    X, Y = np.meshgrid(xs, ys)
    d = f(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)).reshape(X.shape)
    I = np.where((d < 0)[..., None], ORANGE, hx("eef4ff"))
    I[np.abs(d) < 0.05] = INK
    save(name, I, w, h)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    build = camera(-128, 34)    # from the front left, looking down a little
    back = camera(118, 24)      # from behind, to see the sound window
    end = camera(-158, 16, center=(-26, 10, 10), span=44)   # close up of the mouthpiece end
    above = camera(-90, 62, pad=0.04)  # nearly top-down, to look inside the open half
    apart = camera(-128, 24, center=(-3, 0, 19), span=72)   # the two halves pulled apart
    gap = 16

    # Build steps. The newest part is bright; earlier parts are pale.
    draw("step-barrel", build, [(BARREL, ORANGE)])
    draw("step-mouthpiece", build, [(minus(BARREL, MOUTH), TAN), (MOUTH, ORANGE)])
    draw("step-ring", build, [(union(BARREL, MOUTH), TAN), (RING, ORANGE)])
    solid = union(BARREL, MOUTH, RING)
    draw("step-chamber", build, [(solid, TAN)], [(CHAMBER, BLUE)])
    draw("step-tunnel", build, [(solid, TAN)], [(CHAMBER, PALEBLUE), (TUNNEL, BLUE)])
    draw("step-window", back, [(solid, TAN)], [(CHAMBER, PALEBLUE), (TUNNEL, PALEBLUE), (WINDOW, BLUE)])

    # The open half, the copy, and the finished whistle
    draw("half-open", above, [(HALF, ORANGE)])
    # Mid-height slice of the half: x -13 to 9, y 6 to 17.5, 40 pixels per mm.
    slice_view("edge-slice", HALF, 6, -13, 9, 6, 17.5, 40)
    draw("flip-in-place", build, [(union(HALF, flipped(HALF, 0)), ORANGE)])
    draw("stacked", build, [(HALF, TAN), (TOP, ORANGE)])
    draw("apart", apart, [(HALF, TAN), (flipped(HALF, 10 + gap), ORANGE)])
    draw("done-front", build, [(WHISTLE, ORANGE)])
    draw("done-back", back, [(WHISTLE, ORANGE)])
    draw("done-end", end, [(WHISTLE, ORANGE)])

    anchors = {
        "step-mouthpiece": {"mouthpiece": project(build, (-20, 7, 10))},
        "step-ring": {"ring": project(build, (15, 0, 10))},
        "step-chamber": {"chamber": project(build, (0, 0, 14))},
        "step-tunnel": {"tunnel": project(build, (-28, 11, 8))},
        "step-window": {"window": project(back, (-2, 19.5, 10))},
        "half-open": {
            "edge": project(above, (0.1, 11, 10)), "chamber": project(above, (2, -5, 2)),
            "air": [project(above, (x, 11, 10)) for x in (-33, -3)],
            "swirl": [project(above, (8.2 * np.cos(np.radians(a)), 8.2 * np.sin(np.radians(a)), 2))
                      for a in range(70, -200, -15)],
        },
        "flip-in-place": {"top": project(build, (0, 0, 10))},
        "stacked": {"seam": project(build, (9, -9.5, 10))},
        "apart": {
            "bottom": project(apart, (9, -9.5, 5)), "top": project(apart, (9, -9.5, 31)),
            "arrow": [project(apart, (-13, -6, 18)), project(apart, (-13, -6, 8))],
        },
        "done-front": {"slot": project(build, (-26, 11, 10))},
        "done-back": {"window": project(back, (-2, 13, 10)), "ring": project(back, (15, 0, 10))},
        "done-end": {"slot": project(end, (-26, 11, 12))},
    }
    (OUT / "anchors.json").write_text(json.dumps(anchors, indent=1))


if __name__ == "__main__":
    main()
