"""Stochastic 2D pass with randomness only in physical quantities.

Noise enters through B's executed motion along the planned route (OU cross-track/yaw drift, speed scale;
the camera is either free-pointing or fixed to the travel direction),
per-frame detection (object pixel size, FOV-edge margin, motion blur, a slowly
varying nuisance such as lighting) and frame delivery. Acquisition is never
drawn directly: it is still the geometry + track/dwell storage gate applied to
those frames. With profile "none" every pass equals collect_designed.sample_pass.
All profile numbers are declared experimental levels, not measured rates.
"""

import json
from math import atan2, cos, exp, floor, log, pi, radians, sin, sqrt, tan
from pathlib import Path
import random

from .collection_design import route_of
from .world import validate_config, visible


PROFILE_PATH = Path(__file__).with_name("noise_profiles.json")


def load_profiles(path=PROFILE_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile(name, profiles=None):
    profiles = profiles or load_profiles()
    if name not in profiles["profiles"]:
        raise ValueError(f"unknown noise profile {name!r}")
    return {"name": name, "shared": profiles["shared"], **profiles["profiles"][name]}


def _sigmoid(x):
    return 1/(1+exp(-x)) if x >= 0 else exp(x)/(1+exp(x))


def _wrap(angle):
    return atan2(sin(angle), cos(angle))


class _OU:
    """Stationary Ornstein-Uhlenbeck state sampled exactly on a fixed step."""

    def __init__(self, rng, sigma, tau, dt):
        self.rng, self.sigma = rng, sigma
        self.a = exp(-dt/tau)
        self.step_sigma = sigma*sqrt(1-self.a*self.a)
        self.value = rng.gauss(0., sigma) if sigma else 0.

    def advance(self):
        if self.sigma:
            self.value = self.a*self.value+self.step_sigma*self.rng.gauss(0., 1.)
        return self.value


def focal_px(config, shared):
    return shared["image_width_px"]/2/tan(radians(config["camera_fov_degrees"])/2)


def detection_probability(*, depth, bearing, bearing_rate, fov, f_px, det, shared, nuisance=0.):
    """P(detect | geometrically visible). Each factor is a declared logistic."""
    if det is None:
        return 1.
    size_px = f_px*shared["object_width_m"]/depth
    margin_deg = (fov/2-abs(bearing))*180/pi
    blur_px = f_px*abs(bearing_rate)*det["exposure_s"]
    p = (det["p_max"]
         * _sigmoid((size_px-det["size_px50"])/det["size_px_scale"])
         * _sigmoid((margin_deg-det["edge_margin_deg50"])/det["edge_margin_scale_deg"])
         * _sigmoid((det["blur_px50"]-blur_px)/det["blur_px_scale"]))
    if nuisance:
        p = min(max(p, 1e-12), 1-1e-12)
        p = _sigmoid(log(p/(1-p))+nuisance)
    return p


def rollout(config, draw, prof, seed, *, keep_frames=True):
    """One noisy realisation of a planned pass. Returns (frames|None, acquired)."""
    rng = random.Random(seed)
    dt = config["dt_seconds"]
    phase = draw["frame_phase_seconds"] % dt
    speed = draw["speed_mps"]
    duration, _ = validate_config({**config, "speed_mps": speed})
    steps = floor((duration-phase)/dt+1e-9)
    walls = config["wall_rectangles"] if draw["walls"] else []
    target = config["object_xy"]
    fov = radians(config["camera_fov_degrees"])
    f_px = focal_px(config, prof["shared"])
    ex, det = prof["execution"], prof["detection"]
    scale = 1.+rng.gauss(0., ex["speed_scale_sigma"]) if ex else 1.
    lateral = _OU(rng, ex["lateral_sigma_m"] if ex else 0., ex["lateral_tau_s"] if ex else 1., dt)
    yaw = _OU(rng, radians(ex["yaw_sigma_deg"]) if ex else 0., ex["yaw_tau_s"] if ex else 1., dt)
    nuisance = _OU(rng, det["nuisance_logit_sigma"] if det else 0., det["nuisance_tau_s"] if det else 1., dt)
    drop = prof["frame_drop_probability"]
    max_miss = prof["track_max_miss_seconds"]
    origin, psi, camera_rel, _ = route_of(config, draw)
    u, n = (cos(psi), sin(psi)), (-sin(psi), cos(psi))
    heading = psi+camera_rel

    def pose_at(t):
        along, cross = speed*scale*t, lateral.value
        return (origin[0]+u[0]*along+n[0]*cross, origin[1]+u[1]*along+n[1]*cross, heading+yaw.value)

    def bearing(pose):
        return _wrap(atan2(target[1]-pose[1], target[0]-pose[0])-pose[2])

    previous_bearing = bearing(pose_at(phase-dt))
    frames, acquired = [] if keep_frames else None, {}
    track_start = last_hit = None
    for index in range(steps+1):
        t = round(phase+index*dt, 10)
        if index:
            lateral.advance(), yaw.advance(), nuisance.advance()
        pose = pose_at(t)
        b = bearing(pose)
        rate = _wrap(b-previous_bearing)/dt
        previous_bearing = b
        seen = visible(pose, target, z_min=config["camera_min_depth_m"], z_max=config["camera_range_m"],
                       fov=fov, walls=walls)
        delivered = not (drop and rng.random() < drop)
        p = 0.
        if seen:
            depth = (target[0]-pose[0])*cos(pose[2])+(target[1]-pose[1])*sin(pose[2])
            p = detection_probability(depth=depth, bearing=b, bearing_rate=rate, fov=fov, f_px=f_px,
                                      det=det, shared=prof["shared"], nuisance=nuisance.value)
        # Draw even when not needed so one random stream maps to one physical pass.
        draw_u = rng.random() if det else 0.
        hit = seen and delivered and draw_u < p
        if hit:
            # A track survives missed/dropped frames up to max_miss seconds.
            if last_hit is None or round((index-last_hit-1)*dt, 10) > max_miss+1e-9:
                track_start = index
            last_hit = index
        elif last_hit is not None and round((index-last_hit)*dt, 10) > max_miss+1e-9:
            track_start = last_hit = None
        dwell = round((index-track_start)*dt, 10) if hit else 0.
        for kind in "NI":
            if hit and dwell >= config["storage_dwell_seconds"][kind] and kind not in acquired:
                acquired[kind] = t
        if keep_frames:
            frames.append(dict(time=t, pose=list(pose), visible=seen, p_detect=round(p, 6),
                               delivered=delivered, detected=hit, contiguous_visible_seconds=dwell,
                               stored=sorted(acquired)))
    return frames, acquired
