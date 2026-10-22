"""Native Tk/Canvas view for the dependency-free 2D simulator.

The renderer consumes engine snapshots only. Geometry is in metres, with the
world y-axis pointing up. The inspector intentionally labels receiver truth
separately from A's estimates; it never writes truth into the estimator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, ttk


SCENARIOS = {
    "first_event": "첫 사건 · 전달과 검증",
    "pass_clear": "경로 통과 · 지식 재사용",
    "pass_occluded": "숨은 가림 · 추정과 실제 차이",
    "ack_loss": "ACK 유실 · 재전송",
}
BG, PANEL, FIELD = "#091321", "#132239", "#0b1829"
INK, MUTED, LINE = "#edf5ff", "#9caec5", "#263d57"
BLUE, TEAL, AMBER = "#66a8ff", "#51dcc4", "#ffc574"
COLORS = {"A": BLUE, "B": TEAL, "P": "#ff9e87", "Q": "#b5a0ff"}
KINDS = {"N": "정상 위치·방향", "C": "현재 위치", "I": "물체 외형"}
FONT = "Malgun Gothic"
FIXED_DT = 1 / 60


class SimulatorWindow:
    """Small native game-style view; scheduling belongs only to this class."""

    def __init__(self, root: tk.Tk, simulator, *, seed: int = 7):
        self.root, self.simulator, self.seed = root, simulator, seed
        self.playing, self.closed = True, False
        self._after = None
        self._last_tick, self._accumulator = time.perf_counter(), 0.0
        self._snapshot = simulator.snapshot()
        self._offset = (0.0, 0.0)
        self._scale, self._world_height = 1.0, 8.0
        root.title("KARCo · 상대 지식 추정 2D 시뮬레이터")
        root.configure(bg=BG)
        width = min(1440, max(1080, root.winfo_screenwidth() - 80))
        height = min(900, max(720, root.winfo_screenheight() - 100))
        root.geometry(f"{width}x{height}")
        root.minsize(1080, 720)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.option_add("*Font", (FONT, 10))
        root.option_add("*TCombobox*Listbox.background", PANEL)
        root.option_add("*TCombobox*Listbox.foreground", INK)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=LINE,
                        foreground=INK, arrowcolor=INK, bordercolor=LINE,
                        lightcolor=LINE, darkcolor=LINE, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)],
                  foreground=[("readonly", INK)])
        style.configure("Vertical.TScrollbar", background=LINE, troughcolor=PANEL,
                        bordercolor=PANEL, arrowcolor=MUTED)
        self.scenario_var = tk.StringVar(value=SCENARIOS[self._snapshot["scenario"]])
        self.seed_var, self.speed_var = tk.StringVar(value=str(seed)), tk.StringVar(value="1×")
        self.notify_var = tk.BooleanVar(value=True)
        self.hidden_var = tk.BooleanVar(value=bool(self._snapshot.get("occlusion")))
        self.inspect_var = tk.StringVar(value="이벤트 대상")
        self.info_vars = {kind: tk.BooleanVar(value=True) for kind in KINDS}
        self.clock_var, self.status_var = tk.StringVar(), tk.StringVar(value="실시간 2D · 미터 좌표 · 실제 메시지 전송/지연")
        self._build()
        root.bind("<space>", self._keyboard_play)
        root.bind("<Right>", self._keyboard_step)
        self._restart_clock()

    def _button(self, parent, text, command, *, primary=False):
        return tk.Button(parent, text=text, command=command, bg=BLUE if primary else LINE,
                         fg=BG if primary else INK, activebackground="#87bdff" if primary else "#355171",
                         activeforeground=BG if primary else INK, relief="flat", bd=0,
                         padx=12, pady=8, cursor="hand2", font=(FONT, 10, "bold"))

    def _label(self, parent, text="", **options):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=INK, **options)

    def _check(self, parent, text, variable, command=None):
        return tk.Checkbutton(parent, text=text, variable=variable, command=command,
                              bg=PANEL, fg=INK, activebackground=PANEL, activeforeground=INK,
                              selectcolor=LINE, relief="flat", bd=0, cursor="hand2")

    def _build(self):
        top = tk.Frame(self.root, bg=PANEL, padx=18, pady=12)
        top.pack(fill="x")
        self._label(top, "KARCo", font=(FONT, 19, "bold")).pack(side="left")
        tk.Label(top, text="  상대 지식 추정", bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 22))
        select = ttk.Combobox(top, textvariable=self.scenario_var, values=list(SCENARIOS.values()),
                              state="readonly", width=28)
        select.pack(side="left", padx=(0, 14))
        select.bind("<<ComboboxSelected>>", lambda _event: self.reset())
        self._label(top, "Seed").pack(side="left", padx=(0, 6))
        seed = tk.Entry(top, textvariable=self.seed_var, width=6, bg=BG, fg=INK,
                        insertbackground=INK, relief="flat", justify="center")
        seed.pack(side="left", ipady=7)
        seed.bind("<Return>", lambda _event: self.reset())
        self._button(top, "다시 시작", self.reset).pack(side="left", padx=8)
        self._button(top, "데이터 저장 ↓", self.export).pack(side="right")

        playback = tk.Frame(self.root, bg=BG, padx=18, pady=10)
        playback.pack(fill="x")
        self.play_button = self._button(playback, "Ⅱ 일시정지", self.toggle_play, primary=True)
        self.play_button.pack(side="left")
        self._button(playback, "한 프레임 ›", self.single_step).pack(side="left", padx=7)
        self._label(playback, "속도").pack(side="left", padx=(10, 7))
        speed = ttk.Combobox(playback, textvariable=self.speed_var,
                             values=("0.5×", "1×", "2×", "4×"), state="readonly", width=5)
        speed.pack(side="left")
        tk.Label(playback, text="Space 재생/정지 · → 한 프레임", bg=BG, fg=MUTED).pack(side="left", padx=18)
        tk.Label(playback, textvariable=self.clock_var, bg=BG, fg=TEAL,
                 font=(FONT, 15, "bold")).pack(side="right")

        controls = tk.Frame(self.root, bg=PANEL, padx=15, pady=10)
        controls.pack(side="bottom", fill="x")
        self._button(controls, "물체 재이동", lambda: self._action("move_target")).pack(side="left", padx=(0, 7))
        self._button(controls, "B 기억 초기화", self.reset_memory).pack(side="left")
        self._check(controls, "초기화 통지", self.notify_var).pack(side="left", padx=(4, 10))
        self._check(controls, "숨은 가림", self.hidden_var,
                    lambda: self._action("set_hidden_occlusion", self.hidden_var.get())).pack(side="left")
        manual = tk.Frame(controls, bg=PANEL)
        manual.pack(side="right")
        self._label(manual, "직접 전송").pack(side="left", padx=5)
        for kind, var in self.info_vars.items():
            self._check(manual, kind, var).pack(side="left")
        self._button(manual, "보내기 →", self.send_info).pack(side="left", padx=(5, 0))

        status = tk.Frame(self.root, bg=BG, padx=18, pady=5)
        status.pack(side="bottom", fill="x")
        tk.Label(status, textvariable=self.status_var, bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        stage = tk.Frame(self.root, bg=BG, padx=14, pady=2)
        stage.pack(fill="both", expand=True)
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(0, weight=1)
        self.world = tk.Canvas(stage, bg=FIELD, highlightthickness=1, highlightbackground=LINE)
        self.world.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.inspector = tk.Canvas(stage, width=310, bg=PANEL, highlightthickness=0)
        self.inspector.grid(row=0, column=1, sticky="nsew")
        scroll = ttk.Scrollbar(stage, orient="vertical", command=self.inspector.yview)
        scroll.grid(row=0, column=2, sticky="ns")
        self.inspector.configure(yscrollcommand=scroll.set, yscrollincrement=24,
                                 scrollregion=(0, 0, 310, 800))
        self.inspector.bind("<MouseWheel>", self._scroll_inspector)
        self.inspector.bind("<Button-4>", lambda _event: self.inspector.yview_scroll(-1, "units"))
        self.inspector.bind("<Button-5>", lambda _event: self.inspector.yview_scroll(1, "units"))
        self.world.bind("<Configure>", lambda _event: self.render())
        self.inspector.bind("<Configure>", lambda _event: self.render())
        chooser = ttk.Combobox(self.inspector, textvariable=self.inspect_var,
                               values=("이벤트 대상", "P", "Q"), state="readonly", width=11)
        self.inspector.create_window(294, 20, window=chooser, anchor="ne", tags="chooser")
        chooser.bind("<<ComboboxSelected>>", lambda _event: self.render())

    def _keyboard_play(self, event):
        if event.widget.winfo_class() not in ("Entry", "TCombobox", "Button", "Checkbutton"):
            self.toggle_play()
            return "break"

    def _keyboard_step(self, event):
        if event.widget.winfo_class() not in ("Entry", "TCombobox"):
            self.single_step()
            return "break"

    def _scroll_inspector(self, event):
        self.inspector.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _restart_clock(self):
        if self._after is not None:
            self.root.after_cancel(self._after)
        self._last_tick, self._accumulator = time.perf_counter(), 0.0
        self._after = self.root.after(0, self._tick)

    def _tick(self):
        self._after = None
        if self.closed:
            return
        now = time.perf_counter()
        elapsed, self._last_tick = min(now - self._last_tick, .25), now
        if self.playing:
            self._accumulator += elapsed * float(self.speed_var.get().replace("×", ""))
            updates = 0
            while self._accumulator >= FIXED_DT and updates < 10:
                self.simulator.step(FIXED_DT)
                self._accumulator -= FIXED_DT
                updates += 1
            if updates == 10:
                self._accumulator = min(self._accumulator, FIXED_DT)
        previous_state = self._snapshot.get("event", {}).get("state")
        self._snapshot = self.simulator.snapshot()
        if self._snapshot.get("event", {}).get("state") == "complete" and previous_state != "complete":
            self.playing = False
        self.play_button.configure(text="Ⅱ 일시정지" if self.playing else "▶ 재생")
        self.render()
        self._after = self.root.after(33, self._tick)

    def toggle_play(self):
        self.playing = not self.playing
        self._accumulator = 0.0
        self._last_tick = time.perf_counter()
        self.play_button.configure(text="Ⅱ 일시정지" if self.playing else "▶ 재생")

    def single_step(self):
        self.playing, self._accumulator = False, 0.0
        self.simulator.step(.05)
        self._snapshot = self.simulator.snapshot()
        self.play_button.configure(text="▶ 재생")
        self.render()

    def reset(self):
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            self.status_var.set("Seed에는 정수를 입력하세요.")
            return
        scenario = next(key for key, value in SCENARIOS.items() if value == self.scenario_var.get())
        self.seed = seed
        self.simulator.reset(scenario=scenario, seed=seed)
        self._snapshot = self.simulator.snapshot()
        self.hidden_var.set(bool(self._snapshot.get("occlusion")))
        self.playing = True
        self.status_var.set("새 에피소드를 시작했습니다. 같은 시드로 결과를 재현할 수 있습니다.")
        self._restart_clock()
        self.render()

    def _action(self, name, *args, **kwargs):
        try:
            getattr(self.simulator, name)(*args, **kwargs)
            self._snapshot = self.simulator.snapshot()
            self.status_var.set("조작을 반영했습니다. 오른쪽 로그에서 관측/통신 결과를 확인하세요.")
            self.render()
        except (ValueError, RuntimeError) as error:
            self.status_var.set(str(error))

    def reset_memory(self):
        self._action("reset_receiver_memory", notify=self.notify_var.get())

    def send_info(self):
        kinds = [kind for kind, var in self.info_vars.items() if var.get()]
        if kinds:
            self._action("send_info", kinds)
        else:
            self.status_var.set("직접 전송할 N/C/I를 하나 이상 선택하세요.")

    def export(self):
        destination = filedialog.asksaveasfilename(
            parent=self.root, title="결정 시점 데이터 저장", defaultextension=".json",
            initialfile=f"karco-{self._snapshot['scenario']}-seed{self.seed}.json",
            filetypes=[("JSON", "*.json")],
        )
        if not destination:
            return
        try:
            payload = {"scenario": self._snapshot["scenario"], "seed": self.seed,
                       "snapshot": self.simulator.snapshot(),
                       "decision_records": self.simulator.export_records()}
            Path(destination).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError, TypeError) as error:
            self.status_var.set(f"저장 실패: {error}")
        else:
            self.status_var.set(f"저장 완료 · {destination}")

    def close(self):
        self.closed = True
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None
        self.root.destroy()

    def _xy(self, x, y):
        return self._offset[0] + x * self._scale, self._offset[1] + (self._world_height - y) * self._scale

    def _text(self, canvas, x, y, text, *, size=11, color=INK, bold=False, anchor="w", **options):
        return canvas.create_text(x, y, text=text, fill=color, anchor=anchor,
                                  font=(FONT, size, "bold" if bold else "normal"), **options)

    def render(self):
        if self.closed or not hasattr(self, "world"):
            return
        snap = self._snapshot
        self.clock_var.set(f"t = {snap['time']:05.1f} s")
        self._draw_world(snap)
        self._draw_inspector(snap)

    def _draw_world(self, snap):
        c = self.world
        c.delete("all")
        width, height = max(100, c.winfo_width()), max(100, c.winfo_height())
        world = snap.get("world", {"width": 12, "height": 8})
        ww, wh = float(world["width"]), float(world["height"])
        self._scale = max(1, min((width - 80) / ww, (height - 112) / wh))
        self._world_height = wh
        self._offset = ((width - ww * self._scale) / 2, 54 + max(0, (height - 112 - wh * self._scale) / 2))
        self._text(c, 20, 25, "LIVE WORLD", size=12, bold=True)
        self._text(c, width - 20, 25, f"{ww:g} × {wh:g} m  ·  Y ↑", color=MUTED, anchor="e")
        left, top = self._xy(0, wh)
        right, bottom = self._xy(ww, 0)
        c.create_rectangle(left, top, right, bottom, fill="#0d1e31", outline="#36506c", width=2)
        for x in range(int(ww) + 1):
            px, _ = self._xy(x, 0)
            c.create_line(px, top, px, bottom, fill="#172d43")
            self._text(c, px, bottom + 12, str(x), size=8, color="#8198b1", anchor="center")
        for y in range(int(wh) + 1):
            _, py = self._xy(0, y)
            c.create_line(left, py, right, py, fill="#172d43")
            self._text(c, left - 12, py, str(y), size=8, color="#8198b1", anchor="e")
        for robot in snap.get("robots", {}).values():
            path = robot.get("path", [])
            if len(path) >= 2:
                points = [coord for point in path for coord in self._xy(*point[:2])]
                c.create_line(*points, fill="#5b7898", width=2, dash=(5, 6))
        for obj in snap.get("objects", []):
            nx, ny = self._xy(obj["normal_x"], obj["normal_y"])
            zone = .64 * self._scale
            color = COLORS.get(obj["id"], AMBER)
            if not obj.get("lookalike"):
                c.create_rectangle(nx - zone, ny - zone, nx + zone, ny + zone,
                                   fill="#162b43", outline="#44617f", width=2)
                self._text(c, nx, ny - zone - 13, f"{obj['site_id']} · {obj['id']}", color=color, bold=True, anchor="center")
            if math.hypot(obj["x"] - obj["normal_x"], obj["y"] - obj["normal_y"]) > .08:
                ghost = .17 * self._scale
                c.create_rectangle(nx - ghost, ny - ghost, nx + ghost, ny + ghost,
                                   outline=color, dash=(4, 4), width=2)
                self._text(c, nx, ny + zone - 9, "정상 자리", size=9, color=MUTED, anchor="center")
            x, y = self._xy(obj["x"], obj["y"])
            radius = max(12, .2 * self._scale)
            c.create_rectangle(x - radius, y - radius, x + radius, y + radius,
                               fill=color, outline="#fff0e8", width=2)
            self._text(c, x, y, obj["id"], size=12, color=BG, bold=True, anchor="center")
            self._text(c, x, y + radius + 15, f"{obj.get('state', '')} · C v{obj.get('version', 1)}", size=9, color=INK, anchor="center")
        for box in snap.get("occluders", []):
            x1, y1 = self._xy(box["min_x"], box["max_y"])
            x2, y2 = self._xy(box["max_x"], box["min_y"])
            c.create_rectangle(x1, y1, x2, y2, fill="#765534", outline=AMBER, width=2, dash=(5, 3))
            self._text(c, (x1 + x2) / 2, y1 - 15, "가림" if box.get("known_to_a") else "가림 · A는 모름",
                       size=9, color=AMBER, anchor="center")
        robots = snap.get("robots", {})
        for name, robot in robots.items():
            self._draw_robot(name, robot)
        for packet in snap.get("network", []):
            x, y = self._xy(packet["x"], packet["y"])
            color = AMBER if packet.get("lost") else COLORS.get(packet.get("from"), BLUE)
            start, end = robots.get(packet.get("from")), robots.get(packet.get("to"))
            if start and end:
                c.create_line(*self._xy(start["x"], start["y"]), *self._xy(end["x"], end["y"]),
                              fill=color, dash=(3, 6), width=1)
            c.create_oval(x - 8, y - 8, x + 8, y + 8, fill=color, outline=INK, width=2)
            self._text(c, x, y - 19, ("× " if packet.get("lost") else "") + packet.get("label", "packet"),
                       size=9, color=color, anchor="center")
        self._text(c, 20, height - 17, "평가용 전체 장면 · 숨은 상태는 A의 추정 입력과 분리", size=9, color=MUTED)

    def _draw_robot(self, name, robot):
        c, color = self.world, COLORS.get(name, BLUE)
        x, y, heading = robot["x"], robot["y"], robot.get("heading", 0)
        px, py = self._xy(x, y)
        if name == "B":
            radius, fov = robot.get("camera_range", 2.2), robot.get("camera_fov", .9 * math.pi)
            arc = [self._xy(x + radius * math.cos(heading - fov / 2 + fov * i / 18),
                            y + radius * math.sin(heading - fov / 2 + fov * i / 18)) for i in range(19)]
            c.create_polygon(px, py, *[value for point in arc for value in point],
                             fill="#1e6a65", stipple="gray25", outline="#377b76")

        def polygon(points, **options):
            translated = [self._xy(x + math.cos(heading) * dx - math.sin(heading) * dy,
                                   y + math.sin(heading) * dx + math.cos(heading) * dy) for dx, dy in points]
            c.create_polygon(*[value for point in translated for value in point], **options)

        for side in (-1, 1):
            sy = side * .24
            polygon([(-.26, sy - .065), (.24, sy - .065), (.24, sy + .065), (-.26, sy + .065)],
                    fill="#06101e", outline="#a7b8ca", width=1)
        polygon([(-.29, -.19), (.28, -.19), (.38, 0), (.28, .19), (-.29, .19)],
                fill=color, outline="#e2f5ff", width=2)
        tx, ty = self._xy(x + .7 * math.cos(heading), y + .7 * math.sin(heading))
        c.create_line(px, py, tx, ty, fill=color, width=3, arrow="last", arrowshape=(9, 11, 4))
        self._text(c, px, py, name, size=12, color=BG, bold=True, anchor="center")
        caption = str(robot.get("state", ""))
        self._text(c, px, py + max(24, self._scale * .36), f"Robot {name}", size=11, color=color, bold=True, anchor="center")
        self._text(c, px, py + max(42, self._scale * .6), caption, size=9, color=INK, anchor="center")

    def _draw_inspector(self, snap):
        c = self.inspector
        c.delete("dynamic")
        content_height = max(800, c.winfo_height())
        c.configure(scrollregion=(0, 0, 310, content_height))
        # The native selector is retained while only drawn items are replaced.
        original = self._text

        def text(x, y, value, **options):
            return original(c, x, y, value, tags="dynamic", **options)

        def line(y):
            c.create_line(18, y, 292, y, fill=LINE, tags="dynamic")

        event = snap.get("event", {})
        target = self.inspect_var.get()
        if target == "이벤트 대상":
            target = event.get("target", "P")
        text(18, 27, "지식 상태", size=12, bold=True)
        text(18, 68, f"A의 추정  →  B의 {target} 지식", size=12, color=BLUE, bold=True)
        estimates = snap.get("beliefs", {}).get(target, {})
        for index, (kind, label) in enumerate(KINDS.items()):
            y = 106 + index * 62
            probability = float(estimates.get(f"b_{kind}", 0))
            text(18, y, f"{kind}  {label}", size=10)
            text(290, y, f"{probability:.1%}", size=14, color=BLUE, bold=True, anchor="e")
            c.create_rectangle(18, y + 17, 290, y + 25, fill=LINE, outline="", tags="dynamic")
            c.create_rectangle(18, y + 17, 18 + 272 * min(1, max(0, probability)), y + 25,
                               fill=BLUE, outline="", tags="dynamic")
        line(273)
        receiver = snap.get("receiver", {})
        text(18, 298, "B의 실제 지식 · 평가용 정답", size=11, color=TEAL, bold=True)
        knowledge = receiver.get("knowledge", {})
        for index, obj in enumerate(("P", "Q")):
            y = 329 + index * 27
            text(18, y, obj, color=COLORS[obj], bold=True)
            for column, kind in enumerate(KINDS):
                valid = bool(knowledge.get(obj, {}).get(kind, False))
                text(57 + column * 81, y, f"{kind} {'보유' if valid else '없음'}", size=10,
                     color=TEAL if valid else MUTED)
        text(18, 386, f"기억 세대  B: {receiver.get('memory_epoch', 0)} / A가 앎: {receiver.get('known_epoch', 0)}", size=9, color=MUTED)
        line(404)
        text(18, 429, str(event.get("state", "대기")), size=11, color=AMBER, bold=True)
        text(18, 449, str(event.get("summary") or event.get("id") or ""), size=10,
             anchor="nw", width=274)
        stats = snap.get("stats", {})
        text(18, 500, f"전송 {stats.get('bytes_sent', 0):,} B   유실 {stats.get('packets_lost', 0)}   재시도 {stats.get('retries', 0)}",
             size=9, color=MUTED)
        line(520)
        text(18, 544, "이벤트 로그", size=11, bold=True)
        available = max(1, int((content_height - 566) / 42))
        for index, log in enumerate(snap.get("logs", [])[-available:]):
            text(18, 568 + index * 42, f"{log.get('time', 0):05.1f}s  {log.get('message', '')}",
                 size=9, color=MUTED, anchor="nw", width=274)


def run_desktop(simulator, *, seed: int = 7, smoke_test: bool = False) -> None:
    """Launch the native UI; smoke mode closes itself after a few frames."""
    root = tk.Tk()
    window = SimulatorWindow(root, simulator, seed=seed)
    callback_errors = []
    if smoke_test:
        def fail_callback(_error_type, error, traceback):
            callback_errors.append((error, traceback))
            if not window.closed:
                window.close()

        # Tk normally only prints callback errors and mainloop still returns.
        # A smoke run must propagate them so its exit status is meaningful.
        root.report_callback_exception = fail_callback
        # Still exercises layout, snapshot rendering, scheduling, and close.
        root.after(400, window.close)
    root.mainloop()
    if callback_errors:
        error, traceback = callback_errors[0]
        raise error.with_traceback(traceback)
