# Crop Video GUI

Crop Video GUI is a cross-platform Tkinter application that lets you batch-trim videos with an interactive OpenCV preview. Pick top, bottom, left, and right boundaries visually, optionally scale to a target resolution, and export the results through ffmpeg (with GPU encoders when available).

## Features
- Batch process common video formats (`.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, `.flv`, `.wmv`).
- Interactive crop picker with horizontal and vertical rulers, live mouse crosshair, and keyboard/mouse shortcuts.
- Optional zoom-to-fit resampling so the crop fills a requested resolution.
- GPU acceleration via NVIDIA NVENC, AMD AMF, or Intel Quick Sync when detected.
- Live progress log and progress bar that updates with ffmpeg runtime.
- Stop/resume safeguards that cancel ffmpeg cleanly and keep the GUI responsive.
- Remembers your last input/output directories between sessions.

## Requirements
- Python 3.10+
- ffmpeg and ffprobe on the system `PATH`
- pip packages from `requirements.txt` (`opencv-python`, `numpy`, `pyinstaller`)

### Optional GPU support
Install the appropriate GPU-enabled ffmpeg build for NVENC, AMF, or Quick Sync if you want hardware accelerated exports.

## Getting Started

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Launch the GUI:

```bash
python crop_gui.py
```

### Crop Picker Controls
- **Mouse left click** – set top boundary.
- **Mouse right click** – set bottom boundary.
- **A key** – set left boundary at the current cursor position.
- **D key** – set right boundary at the current cursor position.
- **Enter** – confirm selection.
- **Esc** – reset and close the selector without saving.

During batch processing you can click **Stop** to halt the current ffmpeg run; the progress bar reflects elapsed encode time in real time. The **Exit** button also stops safely if encoding is underway.

## Testing

A lightweight regression test guards the ffmpeg filter builder:

```bash
python -m unittest tests.test_build_filter_string
```

Run this before committing changes to the crop logic or target-resolution handling.

## Building Standalone Packages

You can create local bundles with PyInstaller:

```bash
python -m PyInstaller --clean --noconfirm --name CropVideoGUI --onefile --windowed crop_gui.py  # Windows
python -m PyInstaller --clean --noconfirm --name CropVideoGUI --windowed crop_gui.py           # macOS
```

- Windows builds produce `dist/CropVideoGUI.exe`. Compress with your preferred archiver.
- macOS builds produce `dist/CropVideoGUI.app`. Use `hdiutil create` to wrap it in a `.dmg` image if desired.

## Automated Releases

The repository includes `.github/workflows/build-release.yml`, a GitHub Actions workflow that:

1. Triggers on tags starting with `v` (for example `v1.2.0`).
2. Builds Windows and macOS bundles via PyInstaller.
3. Packages Windows output into `CropVideoGUI-windows.zip` and macOS output into `CropVideoGUI-macos.dmg`.
4. Publishes these artifacts as assets on the corresponding GitHub release.

To cut a new release:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

GitHub Actions will handle the rest, and the release will appear under your repo’s **Releases** tab with downloadable binaries.

## Contributing

Pull requests are welcome! If you extend the cropper or workflow, please:
- Update or add tests.
- Document new controls or options in this README.
- Keep the GUI responsive by avoiding long-running work on the main thread.

Enjoy faster, cleaner video crops!
