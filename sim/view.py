"""Watch the 2D world: A patrols, B drives out, instructs the worker, waits, and drives back.

python view.py                                   # Full Sharing, condition base, seed 0
python view.py --policy proposed --condition occluded --seed 3 --speed 5
Policies: full | minimal | proposed | unc (proposed/unc need models from: python ../rl/train.py --env sim2d)
"""
import argparse
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation, font_manager, patches
from env2d import FULL, Sim2DEnv
from world import CONDITIONS, LAYOUT, ellipse_pose, obs_points, routes

_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in _installed),
                                   "DejaVu Sans")  # needs a Korean font for the instruction text
plt.rcParams["axes.unicode_minus"] = False
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--policy", default="full", choices=["full", "minimal", "proposed", "unc"])
p.add_argument("--condition", default="base", choices=list(CONDITIONS))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--speed", type=int, default=3, help="simulation ticks (0.2 s each) per frame")
p.add_argument("--alpha", type=float, default=0.1)
args = p.parse_args()

belief = args.policy != "unc"
env = Sim2DEnv(args.condition, belief=belief, alpha=args.alpha, trace=True)
if args.policy in ("proposed", "unc"):
    from stable_baselines3 import PPO
    name = f"sim2d_{args.policy}_a{str(args.alpha).replace('.', 'p')}"
    model = PPO.load(Path(__file__).resolve().parent.parent / "rl" / "models" / name)
    policy = lambda o: int(model.predict(o, deterministic=True)[0])
else:
    policy = {"full": lambda o: FULL, "minimal": lambda o: 1}[args.policy]
env.reset(seed=args.seed)
w = env.world

# ----- static scene -----
fig = plt.figure(figsize=(13, 6.5))
ax = fig.add_axes([0.03, 0.06, 0.55, 0.88])
lg = fig.add_axes([0.60, 0.06, 0.38, 0.88])
lg.axis("off")
W, H = LAYOUT["size"]
ax.set(xlim=(-0.05, W + 0.05), ylim=(-0.05, H + 0.05), aspect="equal",
       title=f"{args.policy} | {args.condition} | seed {args.seed}")
ax.add_patch(patches.Rectangle((0, 0), W, H, fill=False, lw=2))
(cx, cy), ex, ey = LAYOUT["ellipse"]
ax.add_patch(patches.Ellipse((cx, cy), 2 * ex, 2 * ey, fill=False, color="#888", lw=3, ls="-"))  # A's line
for k in range(len(LAYOUT["spots"])):  # B's lines (trunk + branch)
    xs, ys = zip(*[f[0] for f in routes(False)[k][0]])
    ax.plot(xs, ys, color="#bbb", lw=3, zorder=0)
for k, (sx, sy) in enumerate(LAYOUT["spots"]):
    t = LAYOUT["tag"]
    ax.add_patch(patches.Rectangle((sx - t / 2, sy - t / 2), t, t, color="black"))
    ax.add_patch(patches.Rectangle((sx - t / 4, sy - t / 4), t / 2, t / 2, color="white"))
    ax.text(sx, sy - 0.07, f"S{k}", ha="center", fontsize=9)
    ax.add_patch(patches.Circle((sx, sy), LAYOUT["spot_r"], fill=False, ls=":", color="#999"))
    pos, _, _ = ellipse_pose(obs_points()[k])
    ax.plot(*pos, "x", color="#555")
if w.occluder:
    (x0, y0), (x1, y1) = LAYOUT["occluder"]
    ax.plot([x0, x1], [y0, y1], color="saddlebrown", lw=6, solid_capstyle="butt")
hx, hy = LAYOUT["home"]
ax.text(hx, hy + 0.06, "B standby", fontsize=8, ha="center")

# ----- moving parts -----
def car_corners(pos, heading):
    L, Wd = LAYOUT["car"]
    c, s = math.cos(heading), math.sin(heading)
    return [(pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy)
            for dx, dy in ((L / 2, Wd / 2), (-L / 2, Wd / 2), (-L / 2, -Wd / 2), (L / 2, -Wd / 2))]

r_max = LAYOUT["depth"][1]
half = math.degrees(LAYOUT["fov"]) / 2
a_car = ax.add_patch(patches.Polygon(car_corners((0, 0), 0), color="royalblue", zorder=5))
b_car = ax.add_patch(patches.Polygon(car_corners((0, 0), 0), color="seagreen", zorder=5))
a_fov = ax.add_patch(patches.Wedge((0, 0), r_max, 0, 1, color="royalblue", alpha=0.08))
b_fov = ax.add_patch(patches.Wedge((0, 0), r_max, 0, 1, color="seagreen", alpha=0.08))
obj_patches = []
for s in w.spots:
    for o in s.objs:
        r = LAYOUT["obj_r"]
        pt = patches.Circle(o.pos, r) if o.shape == "o" else patches.Rectangle(o.pos - r, 2 * r, 2 * r)
        obj_patches.append((o, ax.add_patch(pt)))
clock = ax.text(0.02, H - 0.04, "", fontsize=10)
log_txt = lg.text(0, 1, "", va="top", fontsize=8.5)
printed = 0


def update(_):
    global printed
    for _ in range(args.speed):
        if w.done:
            break
        while w.pending:
            w.decide(policy(env.obs()))
        w.tick()
    pos, heading, cam = ellipse_pose(w.a_phi)
    a_car.set_xy(car_corners(pos, heading))
    a_fov.set_center(pos)
    a_fov.set_theta1(math.degrees(cam) - half)
    a_fov.set_theta2(math.degrees(cam) + half)
    bpos, bh = w.b_pose
    b_car.set_xy(car_corners(bpos, bh))
    b_fov.set_center(bpos)
    b_fov.set_theta1(math.degrees(bh) - half)
    b_fov.set_theta2(math.degrees(bh) + half)
    for o, pt in obj_patches:
        pt.set_facecolor(o.color)
        pt.set_edgecolor("black" if o.deviates() else o.color)
        pt.set_hatch("////" if o.toppled else "")
        if o.shape == "o":
            pt.set_center(o.pos)
        else:
            pt.set_xy(o.pos - LAYOUT["obj_r"])
    s = w.stats
    clock.set_text(f"t = {w.t:5.1f} s   B: {w.b_state}   fixed {s['fixed']}/{s['anomalies']}   "
                   f"{s['bytes'] / 1000:.1f} KB, {s['msgs']} msgs")
    for line in w.log[printed:]:
        print(line)
    printed = len(w.log)
    log_txt.set_text("\n".join(w.log[-42:]))
    return []


anim = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False,
                               frames=lambda: iter(lambda: w.done, True))
plt.show()
