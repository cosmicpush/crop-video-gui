#!/usr/bin/env python3
"""
crop_gui.py

A GUI application for batch video cropping with Y-axis ruler selection.
Processes multiple video files (mp4, mkv, webm) from an input directory,
allowing manual crop selection for each file.

Requirements:
  - ffmpeg and ffprobe must be installed and available in your system's PATH.
  - Python packages: opencv-python, numpy
    Install with: pip install -r requirements.txt
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import subprocess
import tempfile
import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: Required packages are not installed.")
    print("Please install with: pip install -r requirements.txt")
    sys.exit(1)


class InteractiveVideoCropper:
    def __init__(self, input_path: str, output_path: str, frame_ts: Optional[float], target_res: Optional[str], progress_callback=None, use_gpu: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.frame_timestamp = frame_ts
        self.target_resolution = target_res
        self.video_width = 0
        self.video_height = 0
        self.y_top: Optional[int] = None
        self.y_bottom: Optional[int] = None
        self.x_left: Optional[int] = None
        self.x_right: Optional[int] = None
        self.current_mouse_y: int = 0
        self.current_mouse_x: int = 0
        self.scale_factor = 1.0
        self.display_img = None
        self.original_img_h = 0
        self.original_img_w = 0
        self.progress_callback = progress_callback
        self.use_gpu = use_gpu
        self.cancel_requested = False
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.total_duration: Optional[float] = None

    @staticmethod
    def detect_gpu_encoders() -> dict:
        """
        Detect available GPU encoders on the system.
        Returns a dict with GPU type and encoder name.
        """
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, check=False
            )

            encoders = {}
            output = result.stdout.lower()

            # Check for NVIDIA NVENC
            if 'h264_nvenc' in output:
                encoders['nvidia'] = 'h264_nvenc'
                logging.info("Detected NVIDIA GPU encoder (NVENC)")

            # Check for AMD AMF
            if 'h264_amf' in output:
                encoders['amd'] = 'h264_amf'
                logging.info("Detected AMD GPU encoder (AMF)")

            # Check for Intel Quick Sync
            if 'h264_qsv' in output:
                encoders['intel'] = 'h264_qsv'
                logging.info("Detected Intel GPU encoder (Quick Sync)")

            return encoders
        except Exception as e:
            logging.warning(f"Could not detect GPU encoders: {e}")
            return {}

    def _resolve_binary(self, binary: str) -> str:
        if os.path.isabs(binary) or os.sep in binary:
            return binary

        path_env = os.environ.get("PATH", "")
        search_paths = path_env.split(os.pathsep) if path_env else []

        # Common locations for Homebrew and system binaries on macOS
        extra_dirs = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        for directory in extra_dirs:
            if directory not in search_paths:
                search_paths.append(directory)

        # Allow binaries bundled alongside the executable (PyInstaller _MEIPASS)
        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            bundle_path = Path(bundle_dir) / binary
            if bundle_path.exists():
                return str(bundle_path)

        resolved = shutil.which(binary, path=os.pathsep.join(search_paths) if search_paths else None)
        if resolved:
            return resolved

        if sys.platform == "win32":
            resolved = shutil.which(f"{binary}.exe", path=os.pathsep.join(search_paths) if search_paths else None)
            if resolved:
                return resolved

        raise FileNotFoundError(binary)

    def _run_command(self, cmd: list) -> subprocess.CompletedProcess:
        prepared_cmd = cmd[:]
        try:
            prepared_cmd[0] = self._resolve_binary(prepared_cmd[0])
        except FileNotFoundError as exc:
            missing = cmd[0]
            message = (
                f"Required dependency '{missing}' was not found. "
                "Install ffmpeg/ffprobe and ensure they are available on the system PATH."
            )
            logging.error(message)
            raise RuntimeError(message) from exc

        return subprocess.run(prepared_cmd, capture_output=True, text=True, check=False)

    def request_cancel(self):
        """Signal any ongoing ffmpeg work to stop."""
        self.cancel_requested = True
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            try:
                self.ffmpeg_process.terminate()
            except Exception as exc:
                logging.warning(f"Failed to terminate ffmpeg process: {exc}")

    def is_running(self) -> bool:
        return self.ffmpeg_process is not None and self.ffmpeg_process.poll() is None

    def get_video_info(self) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
        cmd_dims = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", self.input_path
        ]
        res_dims = self._run_command(cmd_dims)
        try:
            w, h = map(int, res_dims.stdout.strip().split(','))
            dimensions = (w, h)
            self.video_width = w
            self.video_height = h
        except (ValueError, TypeError):
            dimensions = None

        cmd_duration = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", self.input_path
        ]
        res_duration = self._run_command(cmd_duration)
        try:
            duration = float(res_duration.stdout.strip())
        except (ValueError, TypeError):
            duration = None

        return duration, dimensions

    def extract_frame(self, timestamp: float, out_path: str) -> bool:
        cmd = [
            "ffmpeg", "-ss", str(timestamp), "-i", self.input_path,
            "-vframes", "1", "-q:v", "2", "-y", out_path
        ]
        res = self._run_command(cmd)
        if res.returncode != 0:
            logging.error(f"ffmpeg failed to extract frame: {res.stderr}")
            return False
        return True

    def _get_screen_height(self) -> int:
        """
        Attempt to get the screen height dynamically.
        Falls back to 900px if detection fails.
        """
        try:
            # Try using tkinter to get screen dimensions
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_height = root.winfo_screenheight()
            root.destroy()
            logging.info(f"Detected screen height: {screen_height}px")
            return screen_height
        except Exception as e:
            logging.warning(f"Could not detect screen height: {e}. Using default 900px.")
            return 900

    def _mouse_callback(self, event, x, y, flags, param):
        # Get canvas dimensions (including rulers)
        if self.display_img is not None:
            canvas_w = self.display_img.shape[1] + 80  # Add vertical ruler width
            canvas_h = self.display_img.shape[0] + 80  # Add horizontal ruler height
        else:
            return

        # Check if click is on close button
        close_btn_size = 30
        close_btn_x = canvas_w - close_btn_size - 10
        close_btn_y = 10

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if close button was clicked
            if (close_btn_x <= x <= close_btn_x + close_btn_size and
                close_btn_y <= y <= close_btn_y + close_btn_size):
                # Signal to close the window
                self.y_top = self.y_bottom = self.x_left = self.x_right = -999
                return

        # Ensure coordinates are within image area (excluding rulers)
        if x >= 80 and y >= 80:  # Account for rulers
            img_x = x - 80
            img_y = y - 80
            if img_x >= 0 and img_y < self.display_img.shape[1] and img_y >= 0 and img_y < self.display_img.shape[0]:
                self.current_mouse_x = img_x
                self.current_mouse_y = img_y

                if event == cv2.EVENT_LBUTTONDOWN:
                    self.y_top = img_y
                elif event == cv2.EVENT_RBUTTONDOWN:
                    self.y_bottom = img_y

    def _draw_ruler_and_guides(self, image):
        h, w = image.shape[:2]
        vertical_ruler_width = 80
        horizontal_ruler_height = 80
        ruler_bg = (20, 20, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX

        vertical_ruler = np.full((h, vertical_ruler_width, 3), ruler_bg, dtype=np.uint8)
        horizontal_ruler = np.full((horizontal_ruler_height, w, 3), ruler_bg, dtype=np.uint8)

        for y_pos in range(0, self.original_img_h, 50):
            display_y = int(y_pos * self.scale_factor)
            if display_y >= h:
                break
            if y_pos % 100 == 0:
                cv2.line(vertical_ruler, (50, display_y), (vertical_ruler_width - 1, display_y), (255, 255, 255), 1)
                cv2.putText(vertical_ruler, str(y_pos), (5, display_y + 5), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.line(vertical_ruler, (65, display_y), (vertical_ruler_width - 1, display_y), (150, 150, 150), 1)

        for x_pos in range(0, self.original_img_w, 50):
            display_x = int(x_pos * self.scale_factor)
            if display_x >= w:
                break
            if x_pos % 100 == 0:
                cv2.line(horizontal_ruler, (display_x, horizontal_ruler_height - 30), (display_x, horizontal_ruler_height - 5), (255, 255, 255), 1)
                cv2.putText(horizontal_ruler, str(x_pos), (display_x + 5, horizontal_ruler_height - 10), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.line(horizontal_ruler, (display_x, horizontal_ruler_height - 20), (display_x, horizontal_ruler_height - 5), (150, 150, 150), 1)

        cv2.line(image, (0, self.current_mouse_y), (w - 1, self.current_mouse_y), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(image, (self.current_mouse_x, 0), (self.current_mouse_x, h - 1), (255, 255, 0), 1, cv2.LINE_AA)

        if self.y_top is not None:
            cv2.line(image, (0, self.y_top), (w - 1, self.y_top), (0, 255, 0), 2)
            y_orig = int(self.y_top / self.scale_factor)
            text_y = max(25, self.y_top - 10)
            cv2.putText(image, f"TOP: {y_orig}", (10, text_y), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"TOP: {y_orig}", (10, text_y), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        if self.y_bottom is not None:
            cv2.line(image, (0, self.y_bottom), (w - 1, self.y_bottom), (0, 0, 255), 2)
            y_orig = int(self.y_bottom / self.scale_factor)
            text_y = min(h - 10, self.y_bottom + 25)
            cv2.putText(image, f"BOTTOM: {y_orig}", (10, text_y), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"BOTTOM: {y_orig}", (10, text_y), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        if self.x_left is not None:
            cv2.line(image, (self.x_left, 0), (self.x_left, h - 1), (0, 255, 0), 2)
            x_orig = int(self.x_left / self.scale_factor)
            text_x = max(10, min(self.x_left + 10, w - 150))
            cv2.putText(image, f"LEFT: {x_orig}", (text_x, 35), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"LEFT: {x_orig}", (text_x, 35), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        if self.x_right is not None:
            cv2.line(image, (self.x_right, 0), (self.x_right, h - 1), (0, 0, 255), 2)
            x_orig = int(self.x_right / self.scale_factor)
            text_x = max(10, min(self.x_right - 140, w - 150))
            cv2.putText(image, f"RIGHT: {x_orig}", (text_x, 65), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"RIGHT: {x_orig}", (text_x, 65), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        top_row = np.hstack([
            np.full((horizontal_ruler_height, vertical_ruler_width, 3), ruler_bg, dtype=np.uint8),
            horizontal_ruler
        ])
        bottom_row = np.hstack([vertical_ruler, image])

        canvas = np.vstack([top_row, bottom_row])

        # Add a close button in the top-right corner
        close_btn_size = 30
        close_btn_x = canvas.shape[1] - close_btn_size - 10
        close_btn_y = 10
        cv2.rectangle(canvas, (close_btn_x, close_btn_y),
                     (close_btn_x + close_btn_size, close_btn_y + close_btn_size),
                     (0, 0, 0), -1)
        cv2.rectangle(canvas, (close_btn_x, close_btn_y),
                     (close_btn_x + close_btn_size, close_btn_y + close_btn_size),
                     (100, 100, 100), 2)
        # Draw X
        cv2.line(canvas, (close_btn_x + 8, close_btn_y + 8),
                (close_btn_x + close_btn_size - 8, close_btn_y + close_btn_size - 8),
                (255, 255, 255), 2)
        cv2.line(canvas, (close_btn_x + close_btn_size - 8, close_btn_y + 8),
                (close_btn_x + 8, close_btn_y + close_btn_size - 8),
                (255, 255, 255), 2)

        instructions = [
            "Click X button or press Q to close",
            "L-click: Set TOP",
            "R-click: Set BOTTOM",
            "A key: Set LEFT (cursor)",
            "D key: Set RIGHT (cursor)",
            "ENTER: Confirm | ESC: Cancel"
        ]

        base_x = vertical_ruler_width + 10
        base_y = horizontal_ruler_height + 30
        for i, text in enumerate(instructions):
            y_pos = base_y + i * 25
            # Make the close instruction more prominent
            if i == 0:
                cv2.putText(canvas, text, (base_x, y_pos), font, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, text, (base_x, y_pos), font, 0.6, (255, 100, 100), 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, text, (base_x, y_pos), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, text, (base_x, y_pos), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def select_crop_boundaries(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        original_img = cv2.imread(image_path)
        if original_img is None:
            logging.error("Failed to load the extracted frame for selection.")
            return None

        self.original_img_h, self.original_img_w = original_img.shape[:2]
        self.scale_factor = 1.0
        self.display_img = None
        self.y_top = None
        self.y_bottom = None
        self.x_left = None
        self.x_right = None
        self.current_mouse_x = 0
        self.current_mouse_y = 0

        # Get screen height dynamically and use 85% of it for the preview
        screen_height = self._get_screen_height()
        max_display_height = int(screen_height * 0.85)
        logging.info(f"Using max display height: {max_display_height}px")

        if self.original_img_h > max_display_height:
            self.scale_factor = max_display_height / self.original_img_h
            disp_w = int(self.original_img_w * self.scale_factor)
            disp_h = int(self.original_img_h * self.scale_factor)
            self.display_img = cv2.resize(original_img, (disp_w, disp_h))
            logging.info(f"Scaled preview to {disp_w}x{disp_h} (scale factor: {self.scale_factor:.3f})")
        else:
            self.display_img = original_img.copy()
            logging.info("No scaling needed - displaying at original size")

        disp_h, disp_w = self.display_img.shape[:2]
        self.current_mouse_x = disp_w // 2
        self.current_mouse_y = disp_h // 2

        window_name = "Select Crop Boundaries"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Make window closable on macOS
        try:
            # On macOS, enable the close button
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            # Try to bring window to front
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            # Immediately turn off topmost so it's not always on top
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
        except:
            pass

        cv2.setMouseCallback(window_name, self._mouse_callback)

        # Add instruction text about closing
        logging.info("Window opened. Press Enter to confirm, Esc to cancel, or close the window with the X button")

        while True:
            frame_with_guides = self._draw_ruler_and_guides(self.display_img.copy())

            cv2.imshow(window_name, frame_with_guides)
            key = cv2.waitKey(50) & 0xFF  # Increased wait time for better responsiveness

            # Check if window was closed by user (cross button on Windows/Linux)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logging.info("Window closed by user")
                self.y_top = self.y_bottom = self.x_left = self.x_right = None
                break

            # Check if close button was clicked (special value -999)
            if self.y_top == -999:
                logging.info("Close button clicked")
                self.y_top = self.y_bottom = self.x_left = self.x_right = None
                break

            if key == 13:  # Enter key
                if None in (self.y_top, self.y_bottom, self.x_left, self.x_right):
                    logging.warning("All four boundaries (TOP, BOTTOM, LEFT, RIGHT) must be set before confirming.")
                else:
                    break
            elif key == 27:  # Escape key
                logging.info("Crop selection cancelled by user (Esc key)")
                self.y_top = self.y_bottom = self.x_left = self.x_right = None
                break
            elif key in (ord('a'), ord('A')):
                self.x_left = self.current_mouse_x
                logging.info(f"Set LEFT boundary to {int(self.x_left / self.scale_factor)}px")
            elif key in (ord('d'), ord('D')):
                self.x_right = self.current_mouse_x
                logging.info(f"Set RIGHT boundary to {int(self.x_right / self.scale_factor)}px")
            elif key == ord('q') or key == ord('Q'):
                # Added Q key as another way to close
                logging.info("Crop selection cancelled by user (Q key)")
                self.y_top = self.y_bottom = self.x_left = self.x_right = None
                break

        # Properly clean up windows
        try:
            cv2.destroyWindow(window_name)
            # Give it time to close properly
            cv2.waitKey(1)
            cv2.waitKey(1)
            # Destroy any remaining windows
            cv2.destroyAllWindows()
        except Exception as e:
            logging.warning(f"Error closing window: {e}")

        if None in (self.y_top, self.y_bottom, self.x_left, self.x_right):
            return None

        y1 = int(self.y_top / self.scale_factor)
        y2 = int(self.y_bottom / self.scale_factor)
        x1 = int(self.x_left / self.scale_factor)
        x2 = int(self.x_right / self.scale_factor)

        final_y_top = max(0, min(y1, y2))
        final_y_bottom = min(self.video_height or self.original_img_h, max(y1, y2))
        final_x_left = max(0, min(x1, x2))
        width_limit = self.video_width or self.original_img_w
        final_x_right = min(width_limit, max(x1, x2))

        final_w = final_x_right - final_x_left
        final_h = final_y_bottom - final_y_top

        if final_w <= 0 or final_h <= 0:
            logging.warning("Invalid crop dimensions selected. Please ensure LEFT < RIGHT and TOP < BOTTOM.")
            return None

        return final_x_left, final_y_top, final_w, final_h

    def _ensure_even(self, value: int) -> int:
        return value if value % 2 == 0 else value - 1

    def build_filter_string(self, x: int, y: int, w: int, h: int) -> str:
        crop_w = self._ensure_even(w)
        crop_h = self._ensure_even(h)
        filters = [f"crop={crop_w}:{crop_h}:{x}:{y}"]

        if self.target_resolution:
            try:
                tw_str, th_str = self.target_resolution.lower().split('x')
                target_w = self._ensure_even(int(tw_str))
                target_h = self._ensure_even(int(th_str))

                # Step 1: Scale up to cover the target area, preserving aspect ratio.
                filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase")
                # Step 2: Crop the result from the center to the exact target dimensions.
                filters.append(f"crop={target_w}:{target_h}")

                logging.info(f"Will zoom and fill to {target_w}x{target_h}")
            except ValueError:
                logging.warning(f"Invalid target resolution format: '{self.target_resolution}'. Ignoring.")

        return ",".join(filters)

    def run(self) -> Tuple[bool, bool]:
        """
        Execute the crop workflow.
        Returns (success, cancelled).
        """
        self.cancel_requested = False

        if not os.path.exists(self.input_path):
            logging.error(f"Input file not found: {self.input_path}")
            return False, False

        try:
            duration, dims = self.get_video_info()
        except RuntimeError as exc:
            logging.error(str(exc))
            if self.progress_callback:
                self.progress_callback(str(exc))
            return False, False
        self.total_duration = duration
        if dims:
            logging.info(f"Source video dimensions: {dims[0]}x{dims[1]}")
        else:
            logging.error("Could not determine video dimensions. Exiting.")
            return False, False

        timestamp = self.frame_timestamp if self.frame_timestamp is not None else (duration / 2.0 if duration else 1.0)

        with tempfile.TemporaryDirectory(prefix="ruler_crop_") as tmpdir:
            frame_path = os.path.join(tmpdir, "preview_frame.jpg")
            logging.info(f"Extracting frame at {timestamp:.2f}s for selection...")
            if not self.extract_frame(timestamp, frame_path):
                return False, False

            roi = self.select_crop_boundaries(frame_path)
            if not roi:
                logging.info("Selection cancelled. Exiting.")
                return False, True

            if self.cancel_requested:
                logging.info("Processing cancelled before encoding began.")
                return False, True

            x, y, w, h = roi
            logging.info(f"Initial Crop Area: (x={x}, y={y}, w={w}, h={h})")
            filter_str = self.build_filter_string(x, y, w, h)
            logging.info(f"Generated ffmpeg filter: '{filter_str}'")

            if self.use_gpu:
                available_encoders = self.detect_gpu_encoders()
                if available_encoders:
                    gpu_type, encoder = next(iter(available_encoders.items()))
                    logging.info(f"Using GPU acceleration: {gpu_type.upper()} ({encoder})")
                    cmd = ["ffmpeg", "-i", self.input_path, "-vf", filter_str,
                           "-c:v", encoder, "-preset", "medium", "-b:v", "10M",
                           "-c:a", "copy", "-y", self.output_path]
                else:
                    logging.warning("GPU acceleration requested but no GPU encoder found. Falling back to CPU.")
                    cmd = ["ffmpeg", "-i", self.input_path, "-vf", filter_str,
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                           "-c:a", "copy", "-y", self.output_path]
            else:
                cmd = ["ffmpeg", "-i", self.input_path, "-vf", filter_str,
                       "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                       "-c:a", "copy", "-y", self.output_path]

            logging.info("Starting ffmpeg process. This may take a while...")

            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            proc = None
            cancelled = False
            success = False

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    startupinfo=startupinfo,
                    bufsize=1,
                )
                self.ffmpeg_process = proc

                if proc.stdout:
                    for line in iter(proc.stdout.readline, ''):
                        if self.cancel_requested:
                            break
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if self.progress_callback:
                            self.progress_callback(stripped)
                        else:
                            print(line, end='')

                if self.cancel_requested:
                    cancelled = True
                    if proc.poll() is None:
                        proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logging.warning("ffmpeg did not terminate gracefully; forcing kill.")
                        proc.kill()
                        proc.wait()
                else:
                    proc.wait()

                success = (proc.returncode == 0) and not cancelled
            finally:
                if proc and proc.stdout:
                    proc.stdout.close()
                self.ffmpeg_process = None

        if cancelled:
            logging.info("Processing cancelled by user.")
            if os.path.exists(self.output_path):
                try:
                    os.remove(self.output_path)
                    logging.info("Removed partial output file.")
                except OSError as exc:
                    logging.warning(f"Could not remove partial output: {exc}")
        elif success:
            logging.info(f"Successfully created final video: {self.output_path}")
        else:
            logging.error("Failed to create the output video.")

        return success, cancelled


class BatchVideoCropperGUI:
    SUPPORTED_FORMATS = ('.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv')
    APP_CONFIG_DIR = Path.home() / ".crop_video_gui"
    SETTINGS_FILE = APP_CONFIG_DIR / "settings.json"

    def __init__(self, root):
        self.root = root
        self.root.title("Batch Video Cropper")
        self.root.geometry("900x750")
        self.root.resizable(True, True)

        # Set window icon
        try:
            icon_path = Path(__file__).parent / "assets" / "logo.png"
            if icon_path.exists():
                self.root.iconphoto(True, tk.PhotoImage(file=str(icon_path)))
        except Exception as e:
            logging.warning(f"Could not load window icon: {e}")

        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.target_resolution = tk.StringVar(value="3240x2160")
        self.use_gpu = tk.BooleanVar(value=False)
        self.video_files: List[Path] = []
        self.current_file_index = 0
        self.is_processing = False
        self.gpu_available = {}
        self.active_cropper: Optional[InteractiveVideoCropper] = None
        self.crop_data: dict = {}  # Store crop boundaries for each video file

        self.setup_ui()
        self.check_gpu_availability()
        self.load_user_settings()

    def setup_ui(self):
        """Create the GUI layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for responsive layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Title
        title_label = ttk.Label(main_frame, text="Batch Video Cropper",
                                font=('Helvetica', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1

        # Input Directory Section
        ttk.Label(main_frame, text="Input Directory:", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)

        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_dir, state='readonly')
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        input_btn = ttk.Button(input_frame, text="Browse...", command=self.browse_input_dir)
        input_btn.grid(row=0, column=1)
        row += 1

        # Files found label
        self.files_label = ttk.Label(main_frame, text="No files loaded", foreground="gray")
        self.files_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1

        # Output Directory Section
        ttk.Label(main_frame, text="Output Directory:", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)

        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, state='readonly')
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        output_btn = ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir)
        output_btn.grid(row=0, column=1)
        row += 1

        # Target Resolution Section
        ttk.Label(main_frame, text="Target Resolution:", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(15, 5))
        row += 1

        res_frame = ttk.Frame(main_frame)
        res_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        self.res_entry = ttk.Entry(res_frame, textvariable=self.target_resolution, width=20)
        self.res_entry.grid(row=0, column=0, sticky=tk.W)

        ttk.Label(res_frame, text="(e.g., 3240x2160)", foreground="gray").grid(
            row=0, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # GPU Acceleration Section
        gpu_frame = ttk.Frame(main_frame)
        gpu_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5, 10))

        self.gpu_checkbox = ttk.Checkbutton(
            gpu_frame,
            text="Use GPU Acceleration (if available)",
            variable=self.use_gpu,
            state='disabled'  # Will be enabled if GPU is detected
        )
        self.gpu_checkbox.grid(row=0, column=0, sticky=tk.W)

        self.gpu_status_label = ttk.Label(gpu_frame, text="Checking GPU...", foreground="gray")
        self.gpu_status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Progress Section
        ttk.Label(main_frame, text="Progress:", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(15, 5))
        row += 1

        self.progress_label = ttk.Label(main_frame, text="Ready to start", foreground="gray")
        self.progress_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate', length=400)
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        row += 1

        # Log/Status Text Area
        ttk.Label(main_frame, text="Status Log:", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)

        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD, state='disabled',
                                bg='#f0f0f0', fg='black', relief=tk.SUNKEN, borderwidth=1,
                                font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        row += 1

        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(10, 0))

        self.save_blueprint_btn = ttk.Button(
            button_frame,
            text="Save Crop Blueprint",
            command=self.save_crop_blueprint
        )
        self.save_blueprint_btn.grid(row=0, column=0, padx=5)

        self.start_btn = ttk.Button(button_frame, text="Start Processing",
                                     command=self.start_processing, style='Accent.TButton')
        self.start_btn.grid(row=0, column=1, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop",
                                    command=self.stop_processing, state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=5)

        self.clear_progress_btn = ttk.Button(
            button_frame,
            text="Clear Progress",
            command=self.clear_progress
        )
        self.clear_progress_btn.grid(row=0, column=3, padx=5)

        ttk.Button(button_frame, text="Exit", command=self.handle_exit).grid(row=0, column=4, padx=5)

    def check_gpu_availability(self):
        """Check if GPU encoders are available"""
        self.gpu_available = InteractiveVideoCropper.detect_gpu_encoders()

        if self.gpu_available:
            gpu_names = ', '.join([f"{k.upper()}" for k in self.gpu_available.keys()])
            self.gpu_status_label.config(text=f"GPU detected: {gpu_names}", foreground="green")
            self.use_gpu.set(True)
            self.gpu_checkbox.config(state='normal')
            self.gpu_checkbox.state(['selected'])
            self.log_message(f"GPU acceleration available: {gpu_names} (auto-enabled)")
        else:
            self.gpu_status_label.config(text="No GPU detected", foreground="orange")
            self.gpu_checkbox.config(state='disabled')
            self.use_gpu.set(False)
            self.log_message("No GPU encoders detected - will use CPU encoding")

    def load_user_settings(self):
        """Restore last used directories from disk, if available."""
        settings_path = self.SETTINGS_FILE
        if not settings_path.exists():
            return

        try:
            with open(settings_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning(f"Could not load saved settings: {exc}")
            return

        input_dir = data.get("input_dir")
        output_dir = data.get("output_dir")

        if input_dir and Path(input_dir).exists():
            self.input_dir.set(input_dir)
            self.scan_video_files()

        if output_dir:
            self.output_dir.set(output_dir)

    def save_user_settings(self):
        """Persist current directories so they can be restored next launch."""
        try:
            self.APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "input_dir": self.input_dir.get(),
                        "output_dir": self.output_dir.get(),
                    },
                    fh,
                    indent=2,
                )
        except OSError as exc:
            logging.warning(f"Could not save settings: {exc}")

    def handle_exit(self):
        """Handle Exit button clicks, cancelling work if required."""
        if self.is_processing:
            if not messagebox.askyesno("Exit", "Processing is currently running. Stop and exit?"):
                return
            self.stop_processing(confirm=False)
            self.root.after(200, self._await_shutdown)
        else:
            self.root.destroy()

    def clear_progress(self):
        """Clear saved progress and crop data"""
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot clear progress while processing videos.")
            return

        if messagebox.askyesno("Clear Progress",
            "This will clear all saved crop selection progress.\n\n"
            "Are you sure you want to continue?"):
            # Clear the progress snapshot file
            try:
                snapshot_path = self.APP_CONFIG_DIR / "crop_progress_snapshot.json"
                if snapshot_path.exists():
                    snapshot_path.unlink()
                    self.log_message("✓ Cleared saved progress")
                else:
                    self.log_message("No saved progress found to clear")
            except Exception as e:
                self.log_message(f"⚠ Could not clear progress: {str(e)}")

            # Clear in-memory crop data
            self.crop_data = {}

    @staticmethod
    def _parse_ffmpeg_time(line: str) -> Optional[float]:
        if "time=" not in line:
            return None
        try:
            time_fragment = line.split("time=", 1)[1].split()[0]
            hours, minutes, seconds = time_fragment.split(":")
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            return total_seconds
        except (ValueError, IndexError):
            return None

    def _await_shutdown(self):
        """Defer closing the window until background work has stopped."""
        if self.active_cropper and self.active_cropper.is_running():
            self.root.after(200, self._await_shutdown)
        else:
            self.root.destroy()

    def log_message(self, message: str):
        """Add a message to the log text area"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        logging.info(message)

    def browse_input_dir(self):
        """Open directory browser for input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir.set(directory)
            self.scan_video_files()
            self.save_user_settings()

    def browse_output_dir(self):
        """Open directory browser for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.save_user_settings()

    def scan_video_files(self):
        """Scan the input directory for supported video files"""
        input_path = Path(self.input_dir.get())
        if not input_path.exists() or not input_path.is_dir():
            self.files_label.config(text="Invalid directory", foreground="red")
            return

        self.video_files = []
        for ext in self.SUPPORTED_FORMATS:
            self.video_files.extend(input_path.glob(f'*{ext}'))
            self.video_files.extend(input_path.glob(f'*{ext.upper()}'))

        # Remove duplicates and sort
        self.video_files = sorted(list(set(self.video_files)))

        if self.video_files:
            self.files_label.config(
                text=f"Found {len(self.video_files)} video file(s)",
                foreground="green"
            )
            self.log_message(f"Found {len(self.video_files)} video files in {input_path}")
            for vf in self.video_files:
                self.log_message(f"  - {vf.name}")
        else:
            self.files_label.config(text="No video files found", foreground="orange")
            self.log_message(f"No video files found in {input_path}")

    def validate_inputs(self, require_output: bool = True) -> bool:
        """Validate all inputs before starting"""
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory")
            return False

        if require_output and not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return False

        if not self.video_files:
            messagebox.showerror("Error", "No video files found in input directory")
            return False

        # Validate target resolution format
        target_res = self.target_resolution.get().strip()
        if target_res:
            try:
                parts = target_res.lower().split('x')
                if len(parts) != 2:
                    raise ValueError
                int(parts[0])
                int(parts[1])
            except ValueError:
                messagebox.showerror("Error", "Invalid target resolution format. Use WxH (e.g., 3240x2160)")
                return False

        # Create output directory if it doesn't exist
        if require_output and self.output_dir.get():
            output_path = Path(self.output_dir.get())
            output_path.mkdir(parents=True, exist_ok=True)

        return True

    def collect_all_crop_boundaries(self, resume_from_index: int = 0, skip_problematic: bool = True) -> bool:
        """
        Collect crop boundaries for all videos before processing.
        Returns True if all boundaries were collected, False if user cancelled.

        Args:
            resume_from_index: Index to resume from (for recovery after failures)
            skip_problematic: If True, skip problematic videos and continue
        """
        # Load existing crop data if resuming
        if resume_from_index > 0 and self.crop_data:
            self.log_message(f"\nResuming from video {resume_from_index + 1}...")
            self.log_message(f"Already have crop data for {len(self.crop_data)} videos")

        total_files = len(self.video_files)
        successful_crops = 0
        skipped_videos = []
        user_cancelled = False

        self.log_message("="*60)
        self.log_message("STEP 1: Collecting crop boundaries for all videos...")
        if skip_problematic:
            self.log_message("Mode: Skip problematic videos and continue")
        self.log_message("="*60)

        for index in range(resume_from_index, total_files):
            video_file = self.video_files[index]
            self.progress_label.config(
                text=f"Selecting crop boundaries {index + 1}/{total_files}: {video_file.name}"
            )
            self.log_message(f"\n[{index + 1}/{total_files}] Select crop boundaries for: {video_file.name}")

            # Update progress bar
            self.progress_bar['value'] = index
            self.root.update()

            # Check if we already have crop data for this video (resuming)
            if str(video_file) in self.crop_data:
                self.log_message(f"✓ Already have crop data for {video_file.name}, skipping...")
                successful_crops += 1
                continue

            # Create a temporary cropper just for boundary selection
            max_retries = 3
            success = False

            for attempt in range(max_retries):
                try:
                    temp_cropper = InteractiveVideoCropper(
                        input_path=str(video_file),
                        output_path="",  # Not needed for boundary selection
                        frame_ts=None,
                        target_res=None,  # Not needed for boundary selection
                        progress_callback=None,
                        use_gpu=False
                    )

                    # Get video info and extract preview frame
                    duration, dims = temp_cropper.get_video_info()
                    if not dims:
                        self.log_message(f"✗ ERROR: Could not determine dimensions for {video_file.name}")
                        break  # No point retrying this

                    src_w, src_h = dims
                    timestamp = duration / 2.0 if duration else 1.0

                    with tempfile.TemporaryDirectory(prefix="crop_preview_") as tmpdir:
                        frame_path = os.path.join(tmpdir, "preview_frame.jpg")

                        if attempt > 0:
                            self.log_message(f"Extracting preview frame at {timestamp:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                        else:
                            self.log_message(f"Extracting preview frame at {timestamp:.2f}s...")

                        if not temp_cropper.extract_frame(timestamp, frame_path):
                            self.log_message(f"✗ ERROR: Could not extract frame from {video_file.name}")
                            if attempt < max_retries - 1:
                                self.log_message(f"Retrying in 2 seconds...")
                                import time
                                time.sleep(2)  # Wait 2 seconds before retry
                                continue
                            break

                        # Let user select crop boundaries
                        roi = temp_cropper.select_crop_boundaries(frame_path)

                        if not roi:
                            self.log_message(f"⚠ Crop selection cancelled for {video_file.name}")
                            user_cancelled = True
                            break  # User cancelled, don't retry

                        # Store the crop data for this file
                        x, y, w, h = roi
                        self.crop_data[str(video_file)] = {
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'source_width': src_w,
                            'source_height': src_h,
                            'duration_seconds': duration,
                        }
                        self.log_message(f"✓ Crop area saved: x={x}, y={y}, w={w}, h={h}")
                        success = True
                        successful_crops += 1
                        break  # Success, move to next video

                except Exception as e:
                    self.log_message(f"✗ ERROR: Failed to process {video_file.name}: {str(e)}")
                    if attempt < max_retries - 1:
                        self.log_message(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                        continue
                    logging.error(f"Error during boundary selection for {video_file.name}", exc_info=True)
                    if isinstance(e, RuntimeError):
                        messagebox.showerror("Missing Dependency", str(e))
                        return False

            # Handle failed attempts
            if not success:
                if user_cancelled:
                    self.log_message("\n" + "="*60)
                    self.log_message("Crop selection cancelled by user")
                    self.log_message("="*60)
                    return False
                elif skip_problematic:
                    skipped_videos.append(video_file.name)
                    self.log_message(f"⚠ Skipping {video_file.name} due to errors")
                    # Save current progress before skipping
                    self._save_progress_snapshot()
                else:
                    self.log_message("\n" + "="*60)
                    self.log_message(f"Failed to process {video_file.name}. Aborting.")
                    self.log_message("="*60)
                    return False

        # Final summary
        self.progress_bar['value'] = total_files
        self.log_message("\n" + "="*60)

        if skipped_videos:
            self.log_message(f"✓ Completed with {successful_crops}/{total_files} videos processed")
            self.log_message(f"⚠ Skipped {len(skipped_videos)} videos:")
            for video in skipped_videos:
                self.log_message(f"  - {video}")

            if not messagebox.askyesno("Videos Skipped",
                f"Successfully processed {successful_crops} videos, but {len(skipped_videos)} videos failed.\n\n"
                "Do you want to continue with the successful videos?\n\n"
                "Click 'Yes' to continue or 'No' to cancel."):
                return False
        else:
            self.log_message(f"✓ All crop boundaries collected for {total_files} videos!")

        self.log_message("="*60)

        # Save final crop data
        self._save_progress_snapshot()
        return True

    def _save_progress_snapshot(self):
        """Save a snapshot of current crop progress to a temporary file"""
        try:
            snapshot_path = self.APP_CONFIG_DIR / "crop_progress_snapshot.json"
            self.APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            # Create a simplified version for quick saving
            snapshot = {
                "input_directory": self.input_dir.get(),
                "output_directory": self.output_dir.get(),
                "target_resolution": self.target_resolution.get(),
                "crop_data": self.crop_data,
                "video_files": [str(f) for f in self.video_files],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)

        except Exception as e:
            logging.warning(f"Could not save progress snapshot: {e}")

    def _load_progress_snapshot(self) -> Optional[int]:
        """Load progress snapshot and return the index to resume from"""
        snapshot_path = self.APP_CONFIG_DIR / "crop_progress_snapshot.json"

        if not snapshot_path.exists():
            return None

        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            # Check if this snapshot matches current input directory
            if snapshot.get("input_directory") != self.input_dir.get():
                return None

            # Load crop data
            self.crop_data = snapshot.get("crop_data", {})

            # Find the first video that doesn't have crop data
            for i, video_file in enumerate(self.video_files):
                if str(video_file) not in self.crop_data:
                    return i

            # All videos have crop data
            return len(self.video_files)

        except Exception as e:
            logging.warning(f"Could not load progress snapshot: {e}")
            return None

    def save_crop_blueprint(self):
        """Collect crop bounds and write them to a JSON blueprint without encoding."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot save a blueprint while processing videos.")
            return

        if not self.validate_inputs(require_output=False):
            return

        if not self.video_files:
            return

        # Check if there's a progress snapshot to resume from
        resume_index = self._load_progress_snapshot()

        if resume_index is not None and resume_index > 0:
            if messagebox.askyesno("Resume Available",
                f"Found crop data for {resume_index} out of {len(self.video_files)} videos.\n\n"
                "Do you want to resume from where you left off?"):
                pass  # Crop data already loaded
            else:
                self.crop_data = {}
                try:
                    (self.APP_CONFIG_DIR / "crop_progress_snapshot.json").unlink()
                except:
                    pass

        # Temporarily lock down inputs to avoid changes mid-collection
        self.save_blueprint_btn.config(state='disabled')
        self.start_btn.config(state='disabled')
        self.input_entry.config(state='disabled')
        self.output_entry.config(state='disabled')
        self.res_entry.config(state='disabled')
        self.clear_progress_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')

        self.progress_bar['maximum'] = len(self.video_files)
        self.progress_bar['value'] = 0

        self.log_message("\n" + "="*60)
        self.log_message("Blueprint mode: capturing crop measurements only...")
        self.log_message("="*60)

        try:
            if not self.collect_all_crop_boundaries(resume_from_index=resume_index or 0):
                self.progress_label.config(text="Blueprint capture cancelled")
                # Clear progress snapshot since user cancelled
                try:
                    (self.APP_CONFIG_DIR / "crop_progress_snapshot.json").unlink()
                except:
                    pass
                return

            input_root = Path(self.input_dir.get())
            blueprint = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "input_directory": str(input_root),
                "target_resolution": self.target_resolution.get().strip() or None,
                "videos": [],
            }

            for video_path_str, crop_info in self.crop_data.items():
                video_path = Path(video_path_str)
                try:
                    rel_path = str(video_path.relative_to(input_root))
                except ValueError:
                    rel_path = video_path.name

                entry = {
                    "file_name": video_path.name,
                    "relative_path": rel_path,
                    "crop": {
                        "x": crop_info['x'],
                        "y": crop_info['y'],
                        "width": crop_info['w'],
                        "height": crop_info['h'],
                    },
                    "source_width": crop_info.get('source_width'),
                    "source_height": crop_info.get('source_height'),
                    "duration_seconds": crop_info.get('duration_seconds'),
                }
                blueprint["videos"].append(entry)

            json_path = input_root / "crop_blueprint.json"
            try:
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(blueprint, fh, indent=2)
            except OSError as exc:
                self.log_message(f"✗ ERROR: Could not write blueprint file: {exc}")
                messagebox.showerror("Save Failed", f"Unable to write blueprint file:\n{exc}")
                return

            self.progress_label.config(text=f"Saved blueprint for {len(self.crop_data)} video(s)")
            self.log_message(f"✓ Crop blueprint saved to {json_path}")
            messagebox.showinfo("Blueprint Saved", f"Crop blueprint saved to:\n{json_path}")

            # Clear progress snapshot after successful save
            try:
                (self.APP_CONFIG_DIR / "crop_progress_snapshot.json").unlink()
            except:
                pass

        finally:
            # Restore UI state
            if not self.is_processing:
                self.start_btn.config(state='normal')
            self.save_blueprint_btn.config(state='normal')
            self.clear_progress_btn.config(state='normal')
            self.input_entry.config(state='readonly')
            self.output_entry.config(state='readonly')
            self.res_entry.config(state='normal')
            if not self.is_processing:
                self.stop_btn.config(state='disabled')

            self.log_message("="*60)
            self.log_message("Blueprint mode complete")
            self.log_message("="*60)

    def start_processing(self):
        """Start the batch processing"""
        if not self.validate_inputs():
            return

        # Check if there's a progress snapshot to resume from
        resume_index = self._load_progress_snapshot()

        if resume_index is not None and resume_index > 0:
            if resume_index >= len(self.video_files):
                # All videos have crop data, just ask if they want to proceed with encoding
                if messagebox.askyesno("Resume Available",
                    f"Found crop data for all {len(self.video_files)} videos.\n\n"
                    "Do you want to proceed directly to encoding?"):
                    self.crop_data = self._load_crop_data_from_snapshot()
                    self.start_encoding_phase()
                    return
                else:
                    # Clear the snapshot and start fresh
                    try:
                        (self.APP_CONFIG_DIR / "crop_progress_snapshot.json").unlink()
                    except:
                        pass
            else:
                # Partial progress found
                if messagebox.askyesno("Resume Available",
                    f"Found crop data for {resume_index} out of {len(self.video_files)} videos.\n\n"
                    "Do you want to resume from where you left off?\n\n"
                    "Click 'Yes' to resume or 'No' to start fresh."):
                    # Already loaded crop data in _load_progress_snapshot
                    pass
                else:
                    # Clear and start fresh
                    self.crop_data = {}
                    try:
                        (self.APP_CONFIG_DIR / "crop_progress_snapshot.json").unlink()
                    except:
                        pass

        self.is_processing = True
        self.current_file_index = 0

        # Update UI state
        self.start_btn.config(state='disabled')
        self.save_blueprint_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.input_entry.config(state='disabled')
        self.output_entry.config(state='disabled')
        self.res_entry.config(state='disabled')

        self.progress_bar['maximum'] = len(self.video_files)
        self.progress_bar['value'] = 0

        self.save_user_settings()

        # First, collect all crop boundaries (with resume capability)
        if not self.collect_all_crop_boundaries(resume_from_index=resume_index or 0):
            # User cancelled or error occurred
            self.finish_processing()
            return

        # Now process all files with the collected boundaries
        self.start_encoding_phase()

    def start_encoding_phase(self):
        """Start the encoding phase after crop boundaries are collected"""
        self.log_message("\n" + "="*60)
        self.log_message("STEP 2: Processing all videos...")
        self.log_message("="*60)
        self.progress_bar['value'] = 0

        # Clear the progress snapshot after successful crop collection
        try:
            (self.APP_CONFIG_DIR / "crop_progress_snapshot.json").unlink()
        except:
            pass

        self.process_next_file()

    def _load_crop_data_from_snapshot(self) -> dict:
        """Load crop data from snapshot file"""
        snapshot_path = self.APP_CONFIG_DIR / "crop_progress_snapshot.json"
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            return snapshot.get("crop_data", {})
        except:
            return {}

    def process_next_file(self):
        """Process the next video file in the queue using pre-collected crop data"""
        if not self.is_processing or self.current_file_index >= len(self.video_files):
            self.finish_processing()
            return

        current_file = self.video_files[self.current_file_index]
        total_files = len(self.video_files)

        self.progress_label.config(
            text=f"Encoding {self.current_file_index + 1}/{total_files}: {current_file.name}"
        )
        self.log_message(f"\n[{self.current_file_index + 1}/{total_files}] Encoding: {current_file.name}")

        # Get the pre-collected crop data for this file
        crop_info = self.crop_data.get(str(current_file))
        if not crop_info:
            self.log_message(f"✗ ERROR: No crop data found for {current_file.name}")
            self.current_file_index += 1
            self.root.after(100, self.process_next_file)
            return

        x = crop_info['x']
        y = crop_info['y']
        w = crop_info['w']
        h = crop_info['h']
        self.log_message(f"Using crop area: x={x}, y={y}, w={w}, h={h}")

        # Generate output filename
        output_file = Path(self.output_dir.get()) / f"{current_file.stem}_cropped{current_file.suffix}"

        # Get target resolution (empty string if not set)
        target_res = self.target_resolution.get().strip() if self.target_resolution.get().strip() else None

        # Update the GUI
        self.root.update()

        success = False
        cancelled = False
        base_index = self.current_file_index
        self.progress_bar['value'] = base_index

        try:
            # Define progress callback for ffmpeg output
            def progress_callback(line):
                # Filter and display relevant ffmpeg output
                if line and ('time=' in line or 'frame=' in line or 'fps=' in line):
                    self.log_message(line)
                    if 'time=' in line and self.active_cropper and self.active_cropper.total_duration:
                        total_duration = self.active_cropper.total_duration
                        if total_duration and total_duration > 0:
                            elapsed = self._parse_ffmpeg_time(line)
                            if elapsed is not None:
                                fraction = max(0.0, min(elapsed / total_duration, 1.0))
                                self.progress_bar['value'] = base_index + fraction
                    self.root.update()  # Update GUI to show real-time progress

            # Create cropper instance with progress callback
            self.active_cropper = InteractiveVideoCropper(
                input_path=str(current_file),
                output_path=str(output_file),
                frame_ts=None,
                target_res=target_res,
                progress_callback=progress_callback,
                use_gpu=self.use_gpu.get()
            )

            # Get video info to set duration for progress tracking
            duration, dims = self.active_cropper.get_video_info()
            self.active_cropper.total_duration = duration

            if not dims:
                self.log_message(f"✗ ERROR: Could not determine video dimensions")
                success = False
            else:
                # Build the filter string using the pre-collected crop data
                filter_str = self.active_cropper.build_filter_string(x, y, w, h)
                self.log_message(f"Generated ffmpeg filter: '{filter_str}'")

                # Build ffmpeg command
                if self.use_gpu.get():
                    available_encoders = InteractiveVideoCropper.detect_gpu_encoders()
                    if available_encoders:
                        gpu_type, encoder = next(iter(available_encoders.items()))
                        self.log_message(f"Using GPU acceleration: {gpu_type.upper()} ({encoder})")
                        cmd = ["ffmpeg", "-i", str(current_file), "-vf", filter_str,
                               "-c:v", encoder, "-preset", "medium", "-b:v", "10M",
                               "-c:a", "copy", "-y", str(output_file)]
                    else:
                        self.log_message("GPU requested but not available, using CPU")
                        cmd = ["ffmpeg", "-i", str(current_file), "-vf", filter_str,
                               "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                               "-c:a", "copy", "-y", str(output_file)]
                else:
                    cmd = ["ffmpeg", "-i", str(current_file), "-vf", filter_str,
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                           "-c:a", "copy", "-y", str(output_file)]

                self.log_message("Starting ffmpeg encoding...")

                # Execute ffmpeg
                startupinfo = None
                if sys.platform == 'win32':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE

                proc = None
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        startupinfo=startupinfo,
                        bufsize=1,
                    )
                    self.active_cropper.ffmpeg_process = proc

                    if proc.stdout:
                        for line in iter(proc.stdout.readline, ''):
                            if self.active_cropper.cancel_requested:
                                break
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if progress_callback:
                                progress_callback(stripped)

                    if self.active_cropper.cancel_requested:
                        cancelled = True
                        if proc.poll() is None:
                            proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logging.warning("ffmpeg did not terminate gracefully; forcing kill.")
                            proc.kill()
                            proc.wait()
                    else:
                        proc.wait()

                    success = (proc.returncode == 0) and not cancelled

                finally:
                    if proc and proc.stdout:
                        proc.stdout.close()
                    self.active_cropper.ffmpeg_process = None

                if cancelled:
                    self.log_message(f"⚠ Cancelled: {current_file.name}")
                    if os.path.exists(str(output_file)):
                        try:
                            os.remove(str(output_file))
                            self.log_message("Removed partial output file.")
                        except OSError as exc:
                            logging.warning(f"Could not remove partial output: {exc}")
                elif success:
                    self.log_message(f"✓ Completed: {current_file.name}")
                else:
                    self.log_message(f"✗ ERROR: Failed to encode {current_file.name}")

        except Exception as e:
            self.log_message(f"✗ ERROR: Failed to process {current_file.name}: {str(e)}")
            logging.error(f"Error processing {current_file.name}", exc_info=True)
            if isinstance(e, RuntimeError):
                messagebox.showerror("Missing Dependency", str(e))
            success = False
            cancelled = False

        finally:
            self.active_cropper = None

        # Update GUI
        self.root.update()

        # Update progress and move to next file
        if not cancelled:
            self.current_file_index += 1
        self.progress_bar['value'] = self.current_file_index

        if not self.is_processing or cancelled:
            self.finish_processing()
            return

        # Schedule next file processing
        if self.is_processing:
            self.root.after(500, self.process_next_file)

    def stop_processing(self, confirm: bool = True):
        """Stop the batch processing"""
        if confirm:
            if not messagebox.askyesno("Confirm Stop", "Are you sure you want to stop processing?"):
                return

        self.is_processing = False
        if self.active_cropper:
            self.active_cropper.request_cancel()

        self.log_message("Processing stop requested by user")
        self.stop_btn.config(state='disabled')

    def finish_processing(self):
        """Clean up after processing is complete or stopped"""
        self.progress_label.config(
            text=f"Completed {self.current_file_index}/{len(self.video_files)} files"
        )

        if self.current_file_index == len(self.video_files):
            self.log_message("\n" + "="*60)
            self.log_message("All files processed successfully!")
            self.log_message("="*60)
            messagebox.showinfo("Complete", f"Successfully processed {self.current_file_index} files!")

        # Reset UI state
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.input_entry.config(state='readonly')
        self.output_entry.config(state='readonly')
        self.res_entry.config(state='normal')
        self.save_blueprint_btn.config(state='normal')
        self.is_processing = False
        self.active_cropper = None


def main():
    root = tk.Tk()

    # Try to set a nice theme
    try:
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern theme
    except:
        pass

    app = BatchVideoCropperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
