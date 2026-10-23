"""Perspective-taking acquisition model: A re-simulates what B could have seen.

The view contains only A-side knowledge: planned route/speed, A's map (object
pose from the approved N item, walls), nominal camera geometry, the declared
storage dwell and, depending on the information tier, B's planned camera
heading and B pose reports. A's execution-uncertainty assumption is a declared
design choice fixed before evaluation; it is NOT the simulator noise profile.
A assumes a crude per-frame detection rate (constant, or ramping with nominal
object pixel size) instead of the true distance/blur/lighting detector; a
separately fitted monotone calibration corrects the remaining mismatch.
"""

from math import atan2, cos, exp, floor, inf, pi, radians, sin, sqrt, tan
import random


A_ASSUMPTION = {
    "lateral_sigma_m": 0.03, "lateral_tau_s": 2.0,
    "yaw_sigma_deg": 2.0, "yaw_tau_s": 1.0,
    "speed_scale_sigma": 0.02,
    # Crude distance-agnostic detector: makes the score graded in dwell length so
    # pose reports do not collapse marginal passes to 0/1. Not the simulator's model.
    "frame_detect_probability": 0.9,
    "notice": "A-side declared prior on B execution/detection; not fitted, not the simulator profile",
}

# Distance-aware variant, fixed before any run with it. A knows the registered
# object width and the nominal stream width; the ramp shape and thresholds are a
# generic "detectors need ~10-30 px" rule of thumb, deliberately unlike the
# simulator's logistic. Linear from 0 at zero_px to frame_detect_probability at full_px.
A_ASSUMPTION_SIZE = {
    **A_ASSUMPTION,
    "size_ramp": {"object_width_m": 0.10, "image_width_px": 1280, "zero_px": 8.0, "full_px": 24.0},
    "notice": "A-side declared prior; per-frame detection ramps with nominal object pixel size; not fitted, not the simulator profile",
}

ASSUMPTIONS = {"constant": A_ASSUMPTION, "size": A_ASSUMPTION_SIZE}


def _detect_probability(assumption, depth, camera):
    p = assumption["frame_detect_probability"]
    ramp = assumption.get("size_ramp")
    if ramp is None:
        return p
    f_px = ramp["image_width_px"]/2/tan(radians(camera["fov_deg"])/2)
    size_px = f_px*ramp["object_width_m"]/depth
    return p*min(1., max(0., (size_px-ramp["zero_px"])/(ramp["full_px"]-ramp["zero_px"])))


def _blocked(start, end, rectangles):
    """Closed segment/AABB intersection (slab clipping); touching blocks sight."""
    for xmin, ymin, xmax, ymax in rectangles:
        low, high = 0., 1.
        for a, b, minimum, maximum in zip(start, end, (xmin, ymin), (xmax, ymax)):
            delta = b-a
            if delta == 0:
                if not minimum <= a <= maximum:
                    break
            else:
                enter, leave = sorted(((minimum-a)/delta, (maximum-a)/delta))
                low, high = max(low, enter), min(high, leave)
                if low > high:
                    break
        else:
            return True
    return False


def _sees(pose, target, camera, walls):
    """Optical depth if the target is geometrically visible, else None."""
    dx, dy = target[0]-pose[0], target[1]-pose[1]
    angle = atan2(sin(atan2(dy, dx)-pose[2]), cos(atan2(dy, dx)-pose[2]))
    depth = dx*cos(pose[2])+dy*sin(pose[2])
    if (camera["z_min_m"] <= depth <= camera["z_max_m"] and abs(angle) <= radians(camera["fov_deg"])/2
            and not _blocked(pose[:2], target, walls)):
        return depth
    return None


def _ou_path(rng, n, sigma, tau, dt):
    if not sigma:
        return [0.]*n
    a = exp(-dt/tau)
    step = sigma*sqrt(1-a*a)
    value, path = rng.gauss(0., sigma), []
    for _ in range(n):
        path.append(value)
        value = a*value+step*rng.gauss(0., 1.)
    return path


def _report_deviation(reports, t, tau):
    """Mean deviation from plan and time to the nearest report (inf if none)."""
    if not reports:
        return (0., 0., 0.), inf
    if t <= reports[0][0]:
        d = reports[0][0]-t
        decay = exp(-d/tau)
        return (reports[0][1], reports[0][2]*decay, reports[0][3]*decay), d
    if t >= reports[-1][0]:
        d = t-reports[-1][0]
        decay = exp(-d/tau)
        return (reports[-1][1], reports[-1][2]*decay, reports[-1][3]*decay), d
    for (t0, *a), (t1, *b) in zip(reports, reports[1:]):
        if t0 <= t <= t1:
            w = (t-t0)/(t1-t0) if t1 > t0 else 0.
            return tuple(x+(y-x)*w for x, y in zip(a, b)), min(t-t0, t1-t)


def _frame(view):
    psi = radians(view["route_direction_deg"])
    return (cos(psi), sin(psi)), (-sin(psi), cos(psi))


def _planned(view, t, heading):
    (ux, uy), _ = _frame(view)
    along = view["planned_speed_mps"]*t
    return (view["route_origin"][0]+ux*along, view["route_origin"][1]+uy*along, heading)


def acquisition_probability(view, *, samples, seed, assumption=A_ASSUMPTION):
    """Monte Carlo P(stored by pass end) for N and I from A's view only."""
    rng = random.Random(seed)
    camera, dwell = view["camera"], view["dwell_seconds"]
    dt, speed = camera["dt_s"], view["planned_speed_mps"]
    duration = view["route_length_m"]/speed
    target, walls = view["object_xy"], view["walls"]
    u, n = _frame(view)
    planned_heading = None if view.get("planned_heading_deg") is None else radians(view["planned_heading_deg"])
    # Reports as deviations from plan in the route frame: (t, d_along, d_cross, d_yaw).
    reports = []
    for t, x, y, yaw in view.get("pose_reports", []):
        if planned_heading is None:
            raise ValueError("pose reports require the planned heading tier")
        px, py, ph = _planned(view, t, planned_heading)
        dx, dy = x-px, y-py
        reports.append((t, dx*u[0]+dy*u[1], dx*n[0]+dy*n[1], atan2(sin(yaw-ph), cos(yaw-ph))))
    lat_tau, yaw_tau = assumption["lateral_tau_s"], assumption["yaw_tau_s"]
    counts = {"N": 0, "I": 0}
    for _ in range(samples):
        phase = rng.random()*dt
        steps = floor((duration-phase)/dt+1e-9)+1
        heading = planned_heading if planned_heading is not None else rng.uniform(-pi, pi)
        scale_error = rng.gauss(0., assumption["speed_scale_sigma"])
        lateral = _ou_path(rng, steps, assumption["lateral_sigma_m"], lat_tau, dt)
        yaw = _ou_path(rng, steps, radians(assumption["yaw_sigma_deg"]), yaw_tau, dt)
        acquired, start = set(), None
        for k in range(steps):
            t = phase+k*dt
            (mx, my, mh), d = _report_deviation(reports, t, lat_tau)
            free_lat = 1. if d == inf else sqrt(1-exp(-2*d/lat_tau))
            free_yaw = 1. if d == inf else sqrt(1-exp(-2*d/yaw_tau))
            px, py, ph = _planned(view, t, heading)
            along = mx+scale_error*speed*(t if d == inf else d)
            cross = my+lateral[k]*free_lat
            pose = (px+u[0]*along+n[0]*cross, py+u[1]*along+n[1]*cross, ph+mh+yaw[k]*free_yaw)
            depth = _sees(pose, target, camera, walls)
            detect = 0. if depth is None else _detect_probability(assumption, depth, camera)
            if depth is not None and (detect >= 1 or rng.random() < detect):
                start = k if start is None else start
                for kind in "NI":
                    if (k-start)*dt+1e-9 >= dwell[kind]:
                        acquired.add(kind)
            else:
                start = None
            if len(acquired) == 2:
                break
        for kind in acquired:
            counts[kind] += 1
    return {kind: counts[kind]/samples for kind in "NI"}
