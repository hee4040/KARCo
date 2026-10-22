"""Deterministic geometry and B memory. This module never calls the estimator.

Ideal point object, exact map pose/identity, rectangular opaque walls, a planar
camera sector and contiguous visible dwell gate. A sweep is an experimental
design, not a claimed distribution of physical camera errors.
"""

from dataclasses import dataclass
from math import atan2, cos, isfinite, radians, sin


def blocked(start, end, rectangles):
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


def visible(pose, target, *, z_min, z_max, fov, walls):
    x, y, yaw = pose
    dx, dy = target[0]-x, target[1]-y
    angle = atan2(sin(atan2(dy, dx)-yaw), cos(atan2(dy, dx)-yaw))
    optical_z = dx*cos(yaw)+dy*sin(yaw)
    return z_min <= optical_z <= z_max and abs(angle) <= fov/2 and not blocked((x, y), target, walls)


def validate_config(config):
    for name in ("speed_mps", "dt_seconds", "camera_range_m", "one_way_delay_seconds",
                 "ack_deadline_seconds", "current_ttl_seconds"):
        x = config[name]
        if type(x) not in (int, float) or not isfinite(x) or x <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < config["camera_fov_degrees"] <= 360:
        raise ValueError("camera FOV must be in (0,360]")
    if not 0 <= config["camera_min_depth_m"] < config["camera_range_m"]:
        raise ValueError("camera depth window must satisfy 0 <= min < max")
    if config["route_x"][0] >= config["route_x"][1]:
        raise ValueError("route must move in positive x")
    duration = (config["route_x"][1]-config["route_x"][0])/config["speed_mps"]
    steps = duration/config["dt_seconds"]
    if abs(steps-round(steps)) > 1e-8 or steps < 1:
        raise ValueError("route duration must be an integer multiple of dt")
    if 2*config["one_way_delay_seconds"] > config["ack_deadline_seconds"]:
        raise ValueError("ACK deadline must cover both one-way delays")
    if config["current_ttl_seconds"] <= config["ack_deadline_seconds"]:
        raise ValueError("TTL must exceed ACK deadline for this experiment")
    if config["channel_model"] != "ideal_bidirectional_line_of_sight_same_walls":
        raise ValueError("unsupported channel model")
    for kind in "NI":
        dwell = config["storage_dwell_seconds"][kind]
        if not isinstance(dwell, (int, float)) or not isfinite(dwell) or dwell <= 0:
            raise ValueError("storage dwell must be positive")
    for rect in config["wall_rectangles"]:
        if len(rect) != 4 or not all(isfinite(v) for v in rect) or rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError("invalid wall rectangle")
    for name in ("object_xy", "sender_xy", "route_x"):
        if len(config[name]) != 2 or not all(isfinite(v) for v in config[name]):
            raise ValueError(f"invalid {name}")
    for sweep in config["sweeps"].values():
        for key in ("offsets_m", "headings_degrees"):
            if not sweep[key] or not all(isfinite(v) for v in sweep[key]):
                raise ValueError(f"invalid sweep {key}")
        for speed in sweep.get("speeds_mps", [config["speed_mps"]]):
            if type(speed) not in (int, float) or not isfinite(speed) or speed <= 0:
                raise ValueError("sweep speeds must be positive")
            n = (config["route_x"][1]-config["route_x"][0])/speed/config["dt_seconds"]
            if n < 1 or abs(n-round(n)) > 1e-8:
                raise ValueError("every sweep route duration must be an integer multiple of dt")
    return duration, round(steps)


@dataclass(frozen=True)
class PassResult:
    frames: list
    acquired_at: dict
    final_pose: tuple
    walls: list


def run_pass(config, offset, heading, with_walls):
    duration, steps = validate_config(config)
    walls = config["wall_rectangles"] if with_walls else []
    frames, acquired = [], {}
    visible_intervals = 0
    for step in range(steps+1):
        t = round(step*config["dt_seconds"], 10)
        pose = (config["route_x"][0]+config["speed_mps"]*t, offset, radians(heading))
        seen = visible(pose, config["object_xy"], z_min=config["camera_min_depth_m"], z_max=config["camera_range_m"],
                       fov=radians(config["camera_fov_degrees"]), walls=walls)
        # Left-endpoint zero-order hold: interval (t-dt,t] counts only when
        # both sampled endpoints are visible. No invented random detection rate.
        visible_intervals = visible_intervals+1 if seen and frames and frames[-1]["visible"] else 0
        dwell = round(visible_intervals*config["dt_seconds"], 10)
        for kind in "NI":
            if seen and dwell >= config["storage_dwell_seconds"][kind] and kind not in acquired:
                acquired[kind] = t
        frames.append({"time": t, "pose": list(pose), "visible": seen,
                       "contiguous_visible_seconds": dwell, "stored": sorted(acquired)})
    return PassResult(frames, acquired, pose, walls)
