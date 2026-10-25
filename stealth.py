"""
stealth.py — Windows 11 Bedroom Stealth Controller for Overnight AI Training.
Automatically requests Administrator privileges so OpenRGB + PawnIO can turn off
internal motherboard, case fan, RAM, and GPU RGB lights.

Run GUI:  python stealth.py
Run CLI:  python stealth.py --now
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import winreg

# ── Auto-Elevate to Administrator (Required for PawnIO / Internal PC RGB) ────
def ensure_admin() -> None:
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        is_admin = False
    if not is_admin:
        script = str(Path(__file__).resolve())
        args = " ".join([f'"{script}"'] + [f'"{a}"' for a in sys.argv[1:]])
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, args, str(Path(__file__).resolve().parent), 1
        )
        sys.exit(0)


# ── Win32 API Constants & 64-bit Signatures ──────────────────────────────────
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

HWND_BROADCAST = 0xFFFF
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
MONITOR_OFF = 2
MONITOR_ON = -1

VK_VOLUME_MUTE = 0xAD
VK_SPACE = 0x20
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
KEYEVENTF_KEYUP = 0x0002

IDLE_PRIORITY_CLASS = 0x00000040
NORMAL_PRIORITY_CLASS = 0x00000020
PROCESS_ALL_ACCESS = 0x1F0FFF
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008

GUID_SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
GUID_PROCTHROTTLEMAX = "bc5038f7-23e0-4960-96da-33abaf5935ec"
GUID_PERFBOOSTMODE = "be337238-0d82-4146-a960-4f3749d470c7"
GUID_COOLINGPOLICY = "94d3a615-a899-4ac5-ae2b-e4d8f634367f"

DWORD_PTR = ctypes.c_size_t

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
ntdll = ctypes.windll.ntdll

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.SetPriorityClass.restype = wintypes.BOOL

kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, DWORD_PTR]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

kernel32.GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(DWORD_PTR),
    ctypes.POINTER(DWORD_PTR),
]
kernel32.GetProcessAffinityMask.restype = wintypes.BOOL

kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
kernel32.SetThreadExecutionState.restype = wintypes.DWORD

ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
ntdll.NtSuspendProcess.restype = wintypes.LONG

ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
ntdll.NtResumeProcess.restype = wintypes.LONG


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def get_cursor_pos() -> tuple[int, int]:
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def wake_key_pressed() -> bool:
    for vk in (VK_SPACE, VK_RETURN, VK_ESCAPE):
        if user32.GetAsyncKeyState(vk) & 0x8000:
            return True
    return False


def toggle_system_mute() -> None:
    user32.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
    user32.keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_KEYUP, 0)


def set_monitors_power(turn_off: bool) -> None:
    mode = MONITOR_OFF if turn_off else MONITOR_ON
    user32.SendMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, mode)
    if not turn_off:
        x, y = get_cursor_pos()
        user32.SetCursorPos(x + 1, y)
        user32.SetCursorPos(x, y)


def run_hidden(cmd: list[str]) -> None:
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
        check=False,
    )


def find_openrgb() -> str | None:
    in_path = shutil.which("OpenRGB") or shutil.which("OpenRGB.exe")
    if in_path:
        return in_path
    candidates = [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "OpenRGB" / "OpenRGB.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "OpenRGB" / "OpenRGB.exe",
        Path("C:/Program Files (x86)/OpenRGB/OpenRGB.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def is_pawnio_installed() -> bool:
    res = subprocess.run(
        ["sc", "query", "PawnIO"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    return res.returncode == 0


class StealthController:
    def __init__(self) -> None:
        self.active = False
        self.muted_by_us = False
        self.saved_lighting: dict[str, int] = {}
        self.train_proc: subprocess.Popen | None = None
        self.log_handle = None
        self.pacer_stop = threading.Event()
        self.pacer_thread: threading.Thread | None = None
        self.gpu_quiet_duty = 0.55
        atexit.register(self.disable_stealth)

    def apply_quiet_cpu(self, max_cpu_pct: int = 80) -> None:
        run_hidden(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", GUID_SUB_PROCESSOR, GUID_PROCTHROTTLEMAX, str(max_cpu_pct)])
        run_hidden(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", GUID_SUB_PROCESSOR, GUID_PERFBOOSTMODE, "0"])
        run_hidden(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", GUID_SUB_PROCESSOR, GUID_COOLINGPOLICY, "0"])
        run_hidden(["powercfg", "/setactive", "SCHEME_CURRENT"])
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    def restore_normal_cpu(self) -> None:
        run_hidden(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", GUID_SUB_PROCESSOR, GUID_PROCTHROTTLEMAX, "100"])
        run_hidden(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", GUID_SUB_PROCESSOR, GUID_PERFBOOSTMODE, "2"])
        run_hidden(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", GUID_SUB_PROCESSOR, GUID_COOLINGPOLICY, "1"])
        run_hidden(["powercfg", "/setactive", "SCHEME_CURRENT"])
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    def blackout_rgb(self) -> None:
        try:
            key_path = r"Software\Microsoft\Lighting"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                for name in ("AmbientLightingEnabled", "Brightness"):
                    try:
                        val, _ = winreg.QueryValueEx(key, name)
                        self.saved_lighting[name] = int(val)
                    except FileNotFoundError:
                        pass
                winreg.SetValueEx(key, "Brightness", 0, winreg.REG_DWORD, 0)
                winreg.SetValueEx(key, "AmbientLightingEnabled", 0, winreg.REG_DWORD, 0)
        except Exception:
            pass

        openrgb = find_openrgb()
        if openrgb:
            threading.Thread(
                target=run_hidden,
                args=([openrgb, "--noautoconnect", "--color", "000000", "--brightness", "0"],),
                daemon=True,
            ).start()

    def restore_rgb(self) -> None:
        if self.saved_lighting:
            try:
                key_path = r"Software\Microsoft\Lighting"
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    for name, val in self.saved_lighting.items():
                        winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, int(val))
            except Exception:
                pass
            self.saved_lighting.clear()

        openrgb = find_openrgb()
        if openrgb:
            threading.Thread(
                target=run_hidden,
                args=([openrgb, "--noautoconnect", "--color", "00AAFF", "--brightness", "100"],),
                daemon=True,
            ).start()

    def start_or_throttle_training(self, rounds: int = 800_000) -> str:
        root = Path(__file__).resolve().parent
        train_script = root / "train.py"
        if not train_script.exists():
            return "train.py not found in folder (Stealth Mode running standalone)"

        if self.train_proc is None or self.train_proc.poll() is not None:
            env = os.environ.copy()
            env["RAYON_NUM_THREADS"] = "4"
            env["OMP_NUM_THREADS"] = "4"
            env["MKL_NUM_THREADS"] = "4"

            log_dir = root / "logs"
            log_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_handle = open(log_dir / f"stealth_train_{stamp}.log", "w", encoding="utf-8")

            cmd = [sys.executable, "-u", str(train_script), "--rounds", str(rounds)]
            latest_ckpt = root / "plump_transformer_latest.pt"
            if latest_ckpt.exists():
                cmd += ["--resume", str(latest_ckpt)]

            self.train_proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                creationflags=IDLE_PRIORITY_CLASS | CREATE_NO_WINDOW,
            )

        self._apply_process_limits(stealth_on=True)
        return f"Training PID {self.train_proc.pid} running in Whisper-Quiet Eco Mode"

    def _apply_process_limits(self, stealth_on: bool) -> None:
        self.pacer_stop.set()
        if self.train_proc is None or self.train_proc.poll() is not None:
            return
        pid = self.train_proc.pid
        handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not handle:
            return
        try:
            if stealth_on:
                kernel32.SetPriorityClass(handle, IDLE_PRIORITY_CLASS)
                kernel32.SetProcessAffinityMask(handle, DWORD_PTR(0x0F))
            else:
                ntdll.NtResumeProcess(handle)
                kernel32.SetPriorityClass(handle, NORMAL_PRIORITY_CLASS)
                system_mask = DWORD_PTR()
                process_mask = DWORD_PTR()
                if kernel32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                    kernel32.SetProcessAffinityMask(handle, system_mask)
        finally:
            kernel32.CloseHandle(handle)

        if stealth_on and self.gpu_quiet_duty < 0.99:
            self.pacer_stop.clear()
            self.pacer_thread = threading.Thread(
                target=self._gpu_pacer_loop, args=(pid,), daemon=True
            )
            self.pacer_thread.start()

    def _gpu_pacer_loop(self, pid: int) -> None:
        handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not handle:
            return
        cycle = 0.080
        try:
            while not self.pacer_stop.is_set():
                duty = max(0.20, min(1.0, self.gpu_quiet_duty))
                if duty >= 0.99:
                    time.sleep(0.25)
                    continue
                run_time = cycle * duty
                sleep_time = cycle * (1.0 - duty)
                ntdll.NtResumeProcess(handle)
                time.sleep(run_time)
                if self.pacer_stop.is_set():
                    break
                ntdll.NtSuspendProcess(handle)
                time.sleep(sleep_time)
        finally:
            ntdll.NtResumeProcess(handle)
            kernel32.CloseHandle(handle)

    def enable_stealth(
        self,
        launch_train: bool = True,
        mute_audio: bool = True,
        blackout: bool = True,
        cpu_cap: int = 80,
        gpu_duty: int = 55,
    ) -> str:
        self.active = True
        self.gpu_quiet_duty = gpu_duty / 100.0
        self.apply_quiet_cpu(max_cpu_pct=cpu_cap)
        if blackout:
            self.blackout_rgb()
        if mute_audio and not self.muted_by_us:
            toggle_system_mute()
            self.muted_by_us = True
        msg = "Stealth Mode Active"
        if launch_train:
            msg = self.start_or_throttle_training()
        if blackout:
            set_monitors_power(turn_off=True)
        return msg

    def disable_stealth(self) -> None:
        if not self.active:
            return
        self.active = False
        self.pacer_stop.set()
        set_monitors_power(turn_off=False)
        self.restore_normal_cpu()
        self.restore_rgb()
        self._apply_process_limits(stealth_on=False)
        if self.muted_by_us:
            toggle_system_mute()
            self.muted_by_us = False


class StealthApp:
    def __init__(self, ctrl: StealthController, start_immediately: bool = False) -> None:
        self.ctrl = ctrl
        self.root = tk.Tk()
        self.root.title("Bedroom Stealth Overnight Controller (Admin)")
        self.root.geometry("520x475")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)

        self.ctrl.restore_normal_cpu()

        self.var_train = tk.BooleanVar(value=True)
        self.var_blackout = tk.BooleanVar(value=True)
        self.var_mute = tk.BooleanVar(value=True)
        self.var_cpu_cap = tk.IntVar(value=80)
        self.var_gpu_duty = tk.IntVar(value=55)
        self.status_var = tk.StringVar(value="Status: Normal Mode (Admin access active for internal PC RGB)")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if start_immediately:
            self.root.after(500, self.start_countdown)

    def _build_ui(self) -> None:
        bg = "#121212"
        fg = "#E0E0E0"
        accent = "#00C853"

        tk.Label(
            self.root,
            text="BEDROOM STEALTH MODE",
            font=("Segoe UI", 15, "bold"),
            bg=bg,
            fg="#FFFFFF",
        ).pack(pady=(12, 2))

        tk.Label(
            self.root,
            text="Zero Lights • Zero Fan Ramp-Up • Wiggle Mouse to Wake",
            font=("Segoe UI", 9),
            bg=bg,
            fg="#9E9E9E",
        ).pack(pady=(0, 10))

        frame = tk.Frame(self.root, bg="#1E1E1E", padx=16, pady=10)
        frame.pack(fill="x", padx=20)

        for text, var in (
            ("Launch / Resume Plump AI Training (train.py)", self.var_train),
            ("Blackout Monitors + Peripherals + Case RGB", self.var_blackout),
            ("Mute Windows Audio Overnight (No 3 AM Pings)", self.var_mute),
        ):
            tk.Checkbutton(
                frame,
                text=text,
                variable=var,
                bg="#1E1E1E",
                fg=fg,
                selectcolor="#2C2C2C",
                activebackground="#1E1E1E",
                activeforeground="#FFFFFF",
                font=("Segoe UI", 10),
            ).pack(anchor="w", pady=2)

        s_frame = tk.Frame(frame, bg="#1E1E1E")
        s_frame.pack(fill="x", pady=(6, 2))
        tk.Label(
            s_frame,
            text="CPU Cap % (<=95% disables loud Turbo Boost):",
            bg="#1E1E1E",
            fg=fg,
            font=("Segoe UI", 9),
        ).pack(anchor="w")
        tk.Scale(
            s_frame,
            from_=50,
            to=95,
            orient="horizontal",
            variable=self.var_cpu_cap,
            bg="#1E1E1E",
            fg=fg,
            highlightthickness=0,
            troughcolor="#333333",
        ).pack(fill="x")

        tk.Label(
            s_frame,
            text="RX 9060 XT Quiet Duty % (55% = 0-RPM Fan Stop):",
            bg="#1E1E1E",
            fg=fg,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 0))
        tk.Scale(
            s_frame,
            from_=25,
            to=100,
            orient="horizontal",
            variable=self.var_gpu_duty,
            bg="#1E1E1E",
            fg=fg,
            highlightthickness=0,
            troughcolor="#333333",
        ).pack(fill="x")

        self.btn_activate = tk.Button(
            self.root,
            text="ACTIVATE STEALTH MODE (10s Grace -> Wiggle Mouse to Wake)",
            command=self.start_countdown,
            bg=accent,
            fg="#000000",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            pady=10,
            cursor="hand2",
        )
        self.btn_activate.pack(fill="x", padx=20, pady=(12, 6))

        btn_row = tk.Frame(self.root, bg=bg)
        btn_row.pack(fill="x", padx=20, pady=(0, 6))

        if not is_pawnio_installed():
            tk.Button(
                btn_row,
                text="1. Install PawnIO Driver (For Case Fans/RAM)",
                command=self.install_pawnio,
                bg="#2C2C2C",
                fg="#FFD54F",
                font=("Segoe UI", 8, "bold"),
                relief="flat",
                pady=4,
            ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        tk.Button(
            btn_row,
            text="Configure Case Fans in OpenRGB (1-Time Setup)",
            command=self.open_openrgb_gui,
            bg="#2C2C2C",
            fg="#80D8FF",
            font=("Segoe UI", 8),
            relief="flat",
            pady=4,
        ).pack(side="left", expand=True, fill="x")

        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9, "bold"),
            bg=bg,
            fg="#80D8FF",
            wraplength=480,
        ).pack(pady=(4, 0))

    def install_pawnio(self) -> None:
        self.status_var.set("Installing PawnIO driver for motherboard/fan RGB headers...")
        subprocess.Popen(["cmd", "/c", "winget install -e --id namazso.PawnIO & pause"])

    def open_openrgb_gui(self) -> None:
        openrgb = find_openrgb()
        if not openrgb:
            subprocess.Popen(
                ["cmd", "/c", "winget install -e --id OpenRGB.OpenRGB --skip-dependencies & pause"]
            )
            return
        self.status_var.set(
            "Opening OpenRGB as Admin: Resize your Motherboard ARGB zones to 30 LEDs, then close it!"
        )
        subprocess.Popen([openrgb])

    def start_countdown(self) -> None:
        if self.ctrl.active:
            self.ctrl.disable_stealth()
            self.btn_activate.configure(
                text="ACTIVATE STEALTH MODE (10s Grace -> Wiggle Mouse to Wake)",
                bg="#00C853",
            )
            self.status_var.set("Restored Normal Mode (100% CPU/GPU Speed & Lights ON)")
            return

        self.btn_activate.configure(state="disabled")
        self._tick_countdown(10)

    def _tick_countdown(self, remaining: int) -> None:
        if remaining > 0:
            self.status_var.set(
                f"Going dark in {remaining}s... Let go of your mouse! (Wiggle in the morning to wake)"
            )
            self.root.after(1000, self._tick_countdown, remaining - 1)
        else:
            self._engage()

    def _engage(self) -> None:
        msg = self.ctrl.enable_stealth(
            launch_train=self.var_train.get(),
            mute_audio=self.var_mute.get(),
            blackout=self.var_blackout.get(),
            cpu_cap=self.var_cpu_cap.get(),
            gpu_duty=self.var_gpu_duty.get(),
        )
        self.btn_activate.configure(
            state="normal",
            text="STEALTH ACTIVE — Wiggle Mouse or Click Here to Wake",
            bg="#FFAB00",
        )
        self.status_var.set(f"{msg} | Waiting for morning mouse wiggle...")
        self.root.after(1500, self._arm_mouse_watcher)

    def _arm_mouse_watcher(self) -> None:
        self.anchor_pos = get_cursor_pos()
        self._poll_wake()

    def _poll_wake(self) -> None:
        if not self.ctrl.active:
            return
        x, y = get_cursor_pos()
        ax, ay = self.anchor_pos
        dist_sq = (x - ax) ** 2 + (y - ay) ** 2

        if dist_sq > 35 * 35 or wake_key_pressed():
            self.ctrl.disable_stealth()
            self.btn_activate.configure(
                text="ACTIVATE STEALTH MODE (10s Grace -> Wiggle Mouse to Wake)",
                bg="#00C853",
            )
            self.status_var.set(
                "Good morning! Stealth Mode disabled & 100% PC speed restored. (Training continues in background)"
            )
            return

        self.root.after(250, self._poll_wake)

    def _on_close(self) -> None:
        self.ctrl.disable_stealth()
        if self.ctrl.train_proc and self.ctrl.train_proc.poll() is None:
            self.ctrl.train_proc.terminate()
        if self.ctrl.log_handle:
            self.ctrl.log_handle.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ensure_admin()
    parser = argparse.ArgumentParser(description="Windows 11 Bedroom Stealth Controller")
    parser.add_argument("--now", action="store_true", help="Start 10s stealth countdown immediately")
    args = parser.parse_args()

    controller = StealthController()
    app = StealthApp(controller, start_immediately=args.now)
    app.run()