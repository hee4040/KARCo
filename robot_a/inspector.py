"""Robot A anomaly inspector, phase 1: one clearly distinct object per spot, D455.

Each spot has an AprilTag (36h11) lying flat next to its object; the tag frame is the spot frame
(x, y on the table, z up). The normal state is taught once, then every frame is compared to it.

python inspector.py teach   # objects in place -> press s to save the normal state
python inspector.py         # patrol mode: prints an A->B message when an anomaly is confirmed
python inspector.py test    # logic self-check, no camera needed
Keys: s save (teach mode), q quit
"""
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

TAG_SIZE = 0.07      # m, edge of the tag's black square -- measure your print, pose scales with it
ROI = 0.30           # m, objects within this radius of a tag belong to that spot
POS_TOL = 0.03       # m, moved farther than this from the taught position -> misplaced
TOPPLE_RATIO = 0.6   # height below 60% of the taught height -> toppled
CONFIRM = 5          # frames an anomaly must persist before it is reported (filters flicker)
MIN_AREA = 400       # px, smaller color blobs are noise
# ponytail: HSV color classes are enough for distinct objects; phase 2 (look-alikes) needs a detector
COLORS = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "blue":   [((95, 120, 50), (130, 255, 255))],
    "yellow": [((20, 120, 80), (35, 255, 255))],
}
NORMAL = Path(__file__).with_name("normal.json")


def to_tag(pts_cam, rvec, tvec):
    """Camera-frame points (N,3) -> tag frame. p_cam = R p_tag + t  =>  p_tag = R^T (p_cam - t)."""
    R = cv2.Rodrigues(rvec)[0]
    return (pts_cam - tvec.reshape(1, 3)) @ R


def compare(normal, obs):
    """normal: {"obj", "pos": [x, y], "height"} taught for one spot.
    obs: {class: {"pos", "height"}} seen inside that spot. Returns anomaly name or None."""
    o = obs.get(normal["obj"])
    if o is None:
        return "wrong_part" if obs else "missing"
    if o["height"] < TOPPLE_RATIO * normal["height"]:  # before misplaced: toppling also shifts the centroid
        return "toppled"
    if np.hypot(*np.subtract(o["pos"], normal["pos"])) > POS_TOL:
        return "misplaced"
    return None  # ponytail: an extra foreign object next to a correct one is ignored for now


def message(tag, normal, obs, anomaly):
    """A->B message, schema from rl/features.py. Every field is filled; the RL controller decides what to send."""
    cls = normal["obj"] if normal["obj"] in obs else next(iter(obs), None)
    cur = obs.get(cls, {})
    return {"spot": tag, "frame": f"tag_{tag}", "anomaly": anomaly, "obj_class": cls,
            "cur_pose": cur.get("pos"), "target_pose": normal["pos"], "conf": cur.get("conf"),
            "t_obs": round(time.time(), 2)}


def find_objects(hsv, depth_m, K):
    """Largest blob per color -> its 3D points in the camera frame."""
    found = {}
    for cls, ranges in COLORS.items():
        mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > MIN_AREA]
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [c], -1, 255, -1)
        v, u = np.nonzero(blob)
        z = depth_m[v, u]
        ok = z > 0.1
        if ok.sum() < 30:
            continue
        u, v, z = u[ok], v[ok], z[ok]
        pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
        found[cls] = dict(pts=pts, contour=c, conf=float(ok.mean()))  # ponytail: valid-depth ratio as confidence
    return found


def per_spot(found, rvec, tvec):
    """Objects within ROI of one tag, expressed in that tag's frame."""
    obs = {}
    for cls, f in found.items():
        p = to_tag(f["pts"], rvec, tvec)
        xy = np.median(p[:, :2], 0)
        if np.hypot(*xy) < ROI:
            obs[cls] = dict(pos=[round(float(x), 3) for x in xy],
                            height=round(float(np.percentile(p[:, 2], 90)), 3), conf=round(f["conf"], 2))
    return obs


def main(teach):
    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs)
    align = rs.align(rs.stream.color)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    s = TAG_SIZE / 2
    tag_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    normal = {int(k): v for k, v in json.loads(NORMAL.read_text()).items()} if NORMAL.exists() else {}
    if not teach and not normal:
        sys.exit("no normal.json yet -- run: python inspector.py teach")
    streak, reported = {}, {}
    try:
        while True:
            fr = align.process(pipe.wait_for_frames())
            img = np.asanyarray(fr.get_color_frame().get_data())
            depth_m = np.asanyarray(fr.get_depth_frame().get_data()) * scale
            found = find_objects(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), depth_m, K)
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            spots = {}
            for c, tid in zip(corners, ids.flatten() if ids is not None else []):
                ok, rvec, tvec = cv2.solvePnP(tag_pts, c.reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                tid = int(tid)
                spots[tid] = per_spot(found, rvec, tvec)
                cv2.drawFrameAxes(img, K, dist, rvec, tvec, TAG_SIZE)
                text = f"tag {tid}: " + ", ".join(f"{k} ({o['pos'][0]*100:.0f},{o['pos'][1]*100:.0f})cm h{o['height']*100:.0f}"
                                                  for k, o in spots[tid].items())
                if not teach and tid in normal:
                    a = compare(normal[tid], spots[tid])
                    last, n = streak.get(tid, (None, 0))
                    n = n + 1 if a == last else 1
                    streak[tid] = (a, n)
                    if n >= CONFIRM and reported.get(tid) != a:  # state held long enough and changed
                        reported[tid] = a
                        if a:
                            print(json.dumps(message(tid, normal[tid], spots[tid], a)), flush=True)
                    text += f"  -> {a or 'normal'}"
                org = tuple(int(x) for x in c.reshape(4, 2)[0])
                cv2.putText(img, text, (org[0], org[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for cls, f in found.items():
                cv2.drawContours(img, [f["contour"]], -1, (255, 255, 255), 2)
            cv2.imshow("teach: s=save q=quit" if teach else "patrol: q=quit", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if teach and key == ord("s") and not spots:
                print("no tag in view -- nothing saved (axes must be drawn on the tag first)")
            elif teach and key == ord("s"):
                for tid, obs in spots.items():
                    if len(obs) == 1:
                        cls, o = next(iter(obs.items()))
                        normal[tid] = dict(obj=cls, pos=o["pos"], height=o["height"])
                    else:
                        print(f"tag {tid}: need exactly 1 object in the spot, saw {list(obs)}")
                NORMAL.write_text(json.dumps(normal, indent=2))
                print("saved", {k: v["obj"] for k, v in normal.items()})
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def selftest():
    n = dict(obj="red", pos=[0.10, 0.0], height=0.06)
    assert compare(n, {"red": dict(pos=[0.11, 0.0], height=0.06)}) is None
    assert compare(n, {"red": dict(pos=[0.15, 0.0], height=0.06)}) == "misplaced"
    assert compare(n, {"red": dict(pos=[0.13, 0.0], height=0.03)}) == "toppled"
    assert compare(n, {"blue": dict(pos=[0.10, 0.0], height=0.06)}) == "wrong_part"
    assert compare(n, {}) == "missing"
    # a point placed in the tag frame must come back unchanged after cam->tag
    rvec, tvec = np.array([0.3, -0.2, 0.1]), np.array([0.05, -0.02, 0.6])
    p_tag = np.array([[0.1, 0.05, 0.02]])
    p_cam = p_tag @ cv2.Rodrigues(rvec)[0].T + tvec
    assert np.allclose(to_tag(p_cam, rvec, tvec), p_tag)
    print("ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    selftest() if arg == "test" else main(teach=arg == "teach")
