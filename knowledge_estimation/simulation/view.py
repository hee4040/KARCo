"""Native Tk 2D replay of collected episodes (no browser/server).

python -m knowledge_estimation.simulation.view knowledge_estimation/data/simulation_v1
"""

import argparse
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from ..measurement import decision_pairs, read_jsonl


def show(root):
    root = Path(root)
    config = json.loads((root/"scenario.json").read_text(encoding="utf-8"))
    traces = list(read_jsonl(root/"decision_runs"/"world_trace.jsonl"))
    rows = list(decision_pairs(root/"replayed"))
    episodes = {t["episode_id"]: t for t in traces}
    decisions = {key: [r for r in rows if r[0]["episode_id"] == key] for key in episodes}
    window = tk.Tk()
    window.title("KARCo | D455 2D knowledge estimation replay")
    window.geometry("1160x760")
    ttk.Label(window, text="D455 기준 2D 재생 · 이상적인 센서 모델 · 실측 데이터 아님", font=("맑은 고딕", 15)).pack(pady=10)
    controls = ttk.Frame(window)
    controls.pack(fill="x", padx=15)
    selected = tk.StringVar(value=traces[0]["episode_id"])
    combo = ttk.Combobox(controls, textvariable=selected, values=list(episodes), state="readonly", width=28)
    combo.pack(side="left")
    playing = [False]
    button = ttk.Button(controls, text="재생 / 정지", command=lambda: playing.__setitem__(0, not playing[0]))
    button.pack(side="left", padx=10)
    ttk.Label(controls, text="초록: 관측 가능   회색: 가림   파랑: A   주황: B").pack(side="left")
    canvas = tk.Canvas(window, bg="#f5f7fb", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=15, pady=10)
    text = tk.StringVar()
    ttk.Label(window, textvariable=text, justify="left", font=("맑은 고딕", 11)).pack(fill="x", padx=20, pady=5)
    clock = tk.DoubleVar(value=0.)
    slider = ttk.Scale(window, variable=clock, from_=0, to=12)
    slider.pack(fill="x", padx=20, pady=10)

    def draw(*_):
        canvas.delete("all")
        trace = episodes[selected.get()]
        record = decisions[selected.get()]
        end = record[-1][0]["time"]
        slider.configure(to=end)
        t = min(clock.get(), end)
        frames = [f for f in trace["frames"] if f["time"] <= t]
        frame = frames[-1] if frames else trace["frames"][0]
        w, h = max(canvas.winfo_width(), 600), max(canvas.winfo_height(), 300)
        scale = min((w-100)/10, (h-70)/10)
        def xy(point):
            return w/2+point[0]*scale, h/2-point[1]*scale
        for n in range(-5, 6):
            canvas.create_line(*xy((n, -5)), *xy((n, 5)), fill="#e2e7ee")
            canvas.create_line(*xy((-5, n)), *xy((5, n)), fill="#e2e7ee")
        for xmin, ymin, xmax, ymax in trace["walls"]:
            canvas.create_rectangle(*xy((xmin, ymax)), *xy((xmax, ymin)), fill="#778397", outline="")
        canvas.create_line(*xy((config["route_x"][0], trace["offset"])),
                           *xy((config["route_x"][1], trace["offset"])), fill="#f3a64b", width=2, dash=(5, 4))
        x, y, yaw = frame["pose"]
        half = math.radians(config["camera_fov_degrees"])/2
        points = []
        for z, sign in ((config["camera_min_depth_m"], -1), (config["camera_range_m"], -1),
                        (config["camera_range_m"], 1), (config["camera_min_depth_m"], 1)):
            lateral = sign*z*math.tan(half)
            points.extend(xy((x+z*math.cos(yaw)-lateral*math.sin(yaw), y+z*math.sin(yaw)+lateral*math.cos(yaw))))
        canvas.create_polygon(*points, fill="#dcf3e5" if frame["visible"] else "#e4e8ef", outline="#a8b4c5")
        for point, color, label in ((config["sender_xy"], "#3877ca", "A"), ((x, y), "#e49a37", "B"),
                                    (config["object_xy"], "#28a875", "Q")):
            px, py = xy(point)
            canvas.create_oval(px-11, py-11, px+11, py+11, fill=color, outline="white", width=2)
            canvas.create_text(px, py-23, text=label, font=("Arial", 14, "bold"))
        eligible = [r for r in record if r[0]["time"] <= t]
        row, before, after, truth = eligible[-1] if eligible else record[0]
        def fmt(value):
            return "미산정" if value is None else f"{value:.3f}"
        estimates = "   ".join(f"{k}: {fmt(row['b_raw']['b_'+k])}" for k in "NCI")
        labels = "   ".join(f"{k}: R={truth[k]['R']} / K={truth[k]['K']}" for k in "NCI")
        canvas.create_text(18, 18, anchor="nw", text=f"{selected.get()}   t={t:.1f}s", font=("Arial", 14, "bold"), fill="#25334a")
        text.set(f"현재 단계: {row['decision_id'].split('/')[-1]}  (표의 값은 최근 결정 시각 {row['time']:.1f}s 기준)\n"
                 f"A에게 보이는 이력으로 계산한 추정값:  {estimates}\n"
                 f"평가 전용 B 실제 상태:  {labels}\n"
                 f"설계 조건: speed={trace.get('speed_mps', config['speed_mps'])}m/s / heading={trace['heading']}° / 벽={bool(trace['walls'])} / 비공개 초기화={trace['hidden_reset']}")

    def change(_=None):
        clock.set(0.)
        draw()
    combo.bind("<<ComboboxSelected>>", change)
    canvas.bind("<Configure>", draw)
    clock.trace_add("write", draw)
    def tick():
        if playing[0]:
            end = decisions[selected.get()][-1][0]["time"]
            clock.set(0. if clock.get() >= end else min(end, clock.get()+config["dt_seconds"]))
        window.after(max(1, round(config["dt_seconds"]*1000)), tick)
    tick()
    window.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    show(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
