"""Generate grouped train/validation/test data from the native 2D engine.

One family shares layout, physical parameters, seed and intervention schedule
across all four scenarios. It must belong to one split in its entirety.
"""

from contextlib import ExitStack
from dataclasses import dataclass, asdict
import hashlib
import json
from math import isfinite, pi
from pathlib import Path
import random

from ...simulator.engine import SCENARIOS, Simulator
from .features import FEATURE_NAMES, extract_features


@dataclass(frozen=True)
class CollectionConfig:
    families: int = 30
    seconds: float = 60.
    sample_interval: float = 1.
    seed: int = 20261022

    def __post_init__(self):
        if isinstance(self.families, bool) or not isinstance(self.families, int) or self.families < 5:
            raise ValueError("families must be an integer >= 5 (at least one family per evaluation split)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("seconds", "sample_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < .05 or abs(value / .05 - round(value / .05)) > 1e-7:
                raise ValueError(f"{name} must be a positive multiple of the simulator's 0.05s tick")
        if self.sample_interval > self.seconds:
            raise ValueError("sample_interval cannot exceed seconds")


def _write(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def family_splits(families, seed):
    order = list(range(families))
    random.Random(seed).shuffle(order)
    n_validation = max(1, families // 5)
    n_test = max(1, families // 5)
    n_train = families - n_validation - n_test
    return {family: "train" if index < n_train else "validation" if index < n_train+n_validation else "test"
            for index, family in enumerate(order)}


def _family(rng, seconds):
    config = {
        "p_delivery": rng.uniform(.65, 1.), "p_ack": rng.uniform(.55, 1.),
        "p_store": rng.uniform(.5, 1.), "q_learn": rng.uniform(.5, 1.),
        "r_crop": rng.uniform(.65, 1.), "network_delay": rng.uniform(.1, .9),
        "robot_speed": rng.uniform(.7, 1.5), "a_speed": rng.uniform(.8, 1.6),
        "work_duration": rng.uniform(.6, 2.),
    }
    scene = {
        "q_x": rng.uniform(5.6, 6.4), "q_y": rng.uniform(3.1, 3.9),
        "pass_x": rng.uniform(4.1, 4.8), "pass_y": rng.uniform(3.1, 3.9),
        "b_start_x": rng.uniform(.6, 1.4), "b_start_y": rng.uniform(.6, 1.4),
        "b_camera_range": rng.uniform(1.4, 3.), "b_camera_fov": rng.uniform(.65*pi, 1.2*pi),
    }
    # Open-loop exploration: interventions do not read hidden labels to choose
    # the next action. Their hidden occurrence/time is NOT a model input.
    interventions = []
    for fraction, action in ((.16, "send"), (.32, "reset"), (.48, "move"),
                             (.64, "appearance"), (.8, "send")):
        interventions.append({
            "time": round(seconds*(fraction+rng.uniform(-.04, .04))/.05)*.05,
            "action": action, "target": rng.choice(("P", "Q")),
            "notify": rng.choice((True, False)),
            "kinds": rng.choice(("N", "C", "I", "NC", "NI", "NCI")),
        })
    return config, scene, interventions


def run_episode(scenario, seed, config, scene, interventions, seconds, sample_interval):
    """Return frozen samples, complete A journal, and privileged final audit."""
    sim = Simulator(scenario, seed=seed, config=config, scene=scene)
    actions = iter(sorted(interventions, key=lambda item: item["time"]))
    pending = next(actions, None)
    interval_ticks = round(sample_interval / .05)
    for target in ("P", "Q"):
        sim.record_sample(target)
    for tick in range(1, round(seconds / .05)+1):
        sim.step(.05)
        # Freeze scheduled samples before same-boundary interventions. A manual
        # send also records its own pre-send sample via the engine.
        if tick % interval_ticks == 0:
            for target in ("P", "Q"):
                sim.record_sample(target)
        while pending is not None and sim.time + 1e-9 >= pending["time"]:
            action, target = pending["action"], pending["target"]
            if action == "send":
                sim.send_info(pending["kinds"], target=target)
            elif action == "reset":
                sim.reset_receiver_memory(notify=pending["notify"])
            elif action == "move":
                sim.move_object(target)
            elif action == "appearance":
                sim.change_appearance(target)
            else:
                raise ValueError(f"unknown intervention: {action}")
            pending = next(actions, None)
    return sim.export_records(), sim.export_observations(), {
        "event": sim.snapshot()["event"], "stats": sim.stats,
    }


def collect(output, options=CollectionConfig(), progress=None):
    output = Path(output)
    # Never silently replace an existing dataset or mix two collection runs.
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "labels").mkdir()
    (output / "observations").mkdir()
    splits = family_splits(options.families, options.seed)
    rng = random.Random(options.seed)
    metrics = {split: {"episodes": 0, "samples": 0, "positive": {k: 0 for k in "NCI"},
                       "brier_sum": {k: 0. for k in "NCI"}} for split in ("train", "validation", "test")}
    episodes = []
    with ExitStack() as stack:
        inputs = {split: stack.enter_context((output / "inputs" / f"{split}.jsonl").open("w", encoding="utf-8")) for split in metrics}
        labels = {split: stack.enter_context((output / "labels" / f"{split}.jsonl").open("w", encoding="utf-8")) for split in metrics}
        audit = stack.enter_context((output / "episodes.jsonl").open("w", encoding="utf-8"))
        for family in range(options.families):
            config, scene, interventions = _family(rng, options.seconds)
            episode_seed = rng.randrange(2**31)
            split, family_id = splits[family], f"f{family:05d}"
            for scenario_index, scenario in enumerate(SCENARIOS):
                episode_id = f"e{family*len(SCENARIOS)+scenario_index:06d}"
                records, history, final = run_episode(scenario, episode_seed, config, scene, interventions,
                                                     options.seconds, options.sample_interval)
                history_path = output / "observations" / f"{episode_id}.jsonl"
                with history_path.open("w", encoding="utf-8") as stream:
                    for event in history:
                        _write(stream, event)
                for index, record in enumerate(records):
                    view = record["a_view"]
                    sample_id = f"{episode_id}:{index:05d}"
                    features = extract_features(history[:view["history_end"]], view)
                    _write(inputs[split], {"sample_id": sample_id, "episode_id": episode_id,
                                          "time": record["time"], "target": record["target"],
                                          "reason": record["reason"], "a_view": view, "features": features})
                    _write(labels[split], {"sample_id": sample_id, "episode_id": episode_id,
                                          "labels": record["labels"], "audit": record["label_audit"]})
                    metric = metrics[split]
                    metric["samples"] += 1
                    for kind in "NCI":
                        y = record["labels"][kind]
                        metric["positive"][kind] += y
                        metric["brier_sum"][kind] += (view["baseline"][f"b_{kind}"]-y)**2
                entry = {"episode_id": episode_id, "family_id": family_id, "split": split}
                episodes.append(entry)
                metrics[split]["episodes"] += 1
                _write(audit, {**entry, "scenario": scenario, "seed": episode_seed,
                               "physical_config": config, "scene": scene,
                               "interventions": interventions, "final": final,
                               "samples": len(records), "observations": len(history)})
            if progress:
                progress(family+1, options.families)
    for metric in metrics.values():
        count = metric["samples"]
        metric["negative"] = {kind: count-metric["positive"][kind] for kind in "NCI"}
        metric["baseline_brier"] = {kind: metric["brier_sum"][kind]/count for kind in "NCI"}
        del metric["brier_sum"]
    checksums = {str(path.relative_to(output)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(output.rglob("*.jsonl"))}
    manifest = {
        "schema_version": 1, "complete": True, "collection": asdict(options),
        "generator": {"version": 2, "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])).replace("\\", "/"):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), Path(__file__).with_name("features.py").resolve(),
                         Path(__file__).resolve().parents[2] / "simulator" / "engine.py",
                         Path(__file__).resolve().parents[2] / "probabilistic" / "estimator.py",
                         Path(__file__).resolve().parents[2] / "probabilistic" / "probabilities.py",
                         Path(__file__).resolve().parents[2] / "probabilistic" / "visibility.py",
                         Path(__file__).resolve().parents[2] / "simulator" / "bridge.py",
                         Path(__file__).resolve().parents[2] / "types.py")}},
        "feature_names": list(FEATURE_NAMES), "episodes": episodes, "statistics": metrics,
        "sha256": checksums,
        "sampling": "periodic plus pre-send; complete A journal addressed by a_view.history_end",
        "split_unit": "family (all four scenarios share one split)",
        "label_definition": "B currently stores usable N/C/I of the physical world's current version",
        "input_contract": "features only; exclude labels, audit, scenario, IDs, seed, interventions, physical config and baseline",
        "limitations": ["Synthetic 2D data; not measured robot performance.",
                        "I usability is sampled, not determined by a real RGB recognizer.",
                        "Rates are simulation assumptions, not measured probabilities.",
                        "Four scripted task flows with randomized local geometry and open-loop interventions.",
                        "N layout version is fixed; no layout replacement or long-term forgetting model.",
                        "Unknown resets and occlusion make some states unidentifiable from A's history.",
                        "Samples include periodic monitoring and pre-send decisions; report both when training."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
