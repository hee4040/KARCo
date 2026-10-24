"""Perspective-taking pass-by acquisition: A re-draws B's planned pass with its own map and camera model.

Replaces the per-(item, site, route) frequency table p_acq. A knows the route it commanded (B's planned poses),
its own map (site positions, the walls it knows about) and B's nominal camera (forward mount: optical axis =
travel heading). It counts the planned frames in which a site is in view and combines them with two measured
sensor rates that do not depend on the route:

  P(stored | pass)        = 1 - (1 - p_frame)^n      n = planned frames with the site in view
  P(right look | stored)  = q_learn                   I only; N is the approved layout, so 1

p_frame is B's per-frame recognition rate while a site is in view, q_learn (pass) the rate at which B holds the
right look of a site it recognised. Both come from the ParameterStore (successes / trials) and are measured
once for the sensor, not per route; a missing one gives None, never a default.
Frames are treated as independent recognition trials; A's map may miss obstacles (it knows no occluder in the
team world), which the separate score calibration (b_cal) has to absorb. An unknown route or site gives None.
"""

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
import hashlib
import json

from ..parameters import MissingMeasurement


@dataclass(frozen=True)
class Camera:
    fov_rad: float
    depth_min_m: float
    depth_max_m: float


def _wrap(a):
    return (a + pi) % (2 * pi) - pi


def _crosses(p, q, a, b):
    """Do segments pq and ab properly cross?"""
    def side(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    return side(a, b, p) * side(a, b, q) < 0 and side(p, q, a) * side(p, q, b) < 0


def in_view(pose, target, camera, walls=()):
    """Is target inside the camera's depth window and field of view at pose (x, y, heading), with no known
    wall segment ((x1, y1), (x2, y2)) on the line of sight?"""
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    return (camera.depth_min_m <= hypot(dx, dy) <= camera.depth_max_m
            and abs(_wrap(atan2(dy, dx) - pose[2])) <= camera.fov_rad / 2
            and not any(_crosses(pose[:2], target, a, b) for a, b in walls))


def planned_frames(plan, target, camera, walls=()):
    return sum(in_view(pose, target, camera, walls) for pose in plan)


def pass_probability(frames, p_frame):
    return 1 - (1 - p_frame) ** frames


class PerspectiveAcquisition:
    """Callable for KnowledgeEstimator(acquisition=...): context -> (p_stored, p_usable, parameter_ids)."""

    def __init__(self, plans, sites, camera, parameters, *, walls=(), frame_condition="in_view",
                 learn_condition="pass"):
        """plans: route_id -> [(x, y, heading), ...] B's planned poses, one per camera frame.
        sites: site_id -> (x, y) from A's map. walls: segments A's map knows."""
        self.camera, self.parameters = camera, parameters
        self.walls = tuple((tuple(a), tuple(b)) for a, b in walls)
        self.frame_condition, self.learn_condition = frame_condition, learn_condition
        self.frames = {(route, site): planned_frames(plan, xy, camera, self.walls)
                       for route, plan in plans.items() for site, xy in sites.items()}
        # The model, not its inputs: plans and site positions are what A feeds it (a new route or speed is a new
        # input to the same estimator, so a fitted score map still applies); camera, known walls and rates are not.
        definition = dict(model="perspective-frames-v1", camera=[camera.fov_rad, camera.depth_min_m, camera.depth_max_m],
                          walls=self.walls, conditions=[frame_condition, learn_condition])
        self.fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()

    def _rate(self, name, kind, condition):
        context = dict(receiver_id="B", kind=kind, site_id="", route_id="", condition_id=condition)
        try:
            entry = self.parameters.entry(name, context)
        except MissingMeasurement:
            return None, ()
        return entry["value"], (entry["parameter_id"],)

    def __call__(self, context):
        frames = self.frames.get((context["route_id"], context["site_id"]))
        if frames is None:
            return None, None, ()
        p_frame, ids = self._rate("p_frame", "", self.frame_condition)
        stored = None if p_frame is None else pass_probability(frames, p_frame)
        if context["kind"] != "I":
            return stored, 1.0, ids
        q_learn, q_ids = self._rate("q_learn", "I", self.learn_condition)
        return stored, q_learn, ids + q_ids
