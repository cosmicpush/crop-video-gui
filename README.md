# Crop Video GUI

<div align="center">
  <img src="assets/logo.png" alt="Crop Video GUI Logo" width="200"/>

  **A powerful cross-platform video cropping tool with batch processing**

  [![Release](https://img.shields.io/github/v/release/cosmicpush/crop-video-gui)](https://github.com/cosmicpush/crop-video-gui/releases)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
</div>

---

## Overview

Crop Video GUI is a professional desktop application for batch video cropping with an interactive visual interface. Select crop boundaries once for each video, then let the app automatically process your entire batch with GPU acceleration support.

### Key Features

- **Two-Phase Workflow** - Select crop boundaries for all videos first, then batch encode without interruption
- **Visual Crop Selection** - Interactive preview with rulers, crosshairs, and real-time boundary markers
- **Batch Processing** - Process multiple videos in one session with consistent settings
- **GPU Acceleration** - Automatic detection and use of NVIDIA NVENC, AMD AMF, or Intel Quick Sync
- **Target Resolution** - Optional zoom-to-fit scaling to any target resolution
- **Multiple Formats** - Supports MP4, MKV, WebM, AVI, MOV, FLV, and WMV
- **Session Memory** - Remembers your last input/output directories
- **Progress Tracking** - Real-time ffmpeg progress with time estimates

## Download

**Get the latest release for your platform:**

- [Windows (EXE)](https://github.com/cosmicpush/crop-video-gui/releases/latest) - Download `CropVideoGUI-windows.zip`
- [macOS (DMG)](https://github.com/cosmicpush/crop-video-gui/releases/latest) - Download `CropVideoGUI-macos.dmg`

### Installation

**Windows:**
1. Download `CropVideoGUI-windows.zip`
2. Extract the ZIP file
3. Run `CropVideoGUI.exe`

**macOS:**
1. Download `CropVideoGUI-macos.dmg`
2. Open the DMG file
3. Drag the app to Applications
4. Right-click and select "Open" (first time only)

**Requirements:**
- **ffmpeg and ffprobe** must be installed on your system
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## How to Use

### Step 1: Select Directories
1. Choose your **Input Directory** containing video files
2. Choose an **Output Directory** for cropped videos
3. (Optional) Set a **Target Resolution** like `1920x1080`

### Step 2: Crop Selection Phase
1. Click **Start Processing**
2. For each video, a preview window will open:
   - **Left-click** → Set top boundary
   - **Right-click** → Set bottom boundary
   - **A key** → Set left boundary
   - **D key** → Set right boundary
   - **Enter** → Confirm and move to next video
   - **Esc** → Cancel

### Step 3: Batch Encoding
- After all crop boundaries are selected, the app automatically encodes all videos
- Monitor progress in the status log
- All videos will be processed with your selected crop areas

### Controls

| Action | Control |
|--------|---------|
| Set Top Boundary | Left Mouse Click |
| Set Bottom Boundary | Right Mouse Click |
| Set Left Boundary | A Key |
| Set Right Boundary | D Key |
| Confirm Selection | Enter |
| Cancel | Esc |
| Stop Processing | Stop Button |

## Development Setup

### Prerequisites
- Python 3.10 or higher
- ffmpeg and ffprobe in system PATH

### Installation

```bash
# Clone the repository
git clone https://github.com/cosmicpush/crop-video-gui.git
cd crop-video-gui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the Application

```bash
python crop_gui.py
```

### Run Tests

```bash
python -m unittest tests.test_build_filter_string
```

## Building from Source

### Local Build

**Windows:**
```bash
python -m PyInstaller --clean --noconfirm --name CropVideoGUI --onefile --windowed --icon=assets/logo.png crop_gui.py
```

**macOS:**
```bash
python -m PyInstaller --clean --noconfirm --name CropVideoGUI --windowed --icon=assets/logo.png crop_gui.py
```

### Automated Releases

The project includes GitHub Actions for automated builds:

1. **Create a version tag:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **GitHub Actions automatically:**
   - Converts `assets/logo.png` to platform-specific icons
   - Builds Windows EXE with embedded icon
   - Builds macOS DMG with custom icon and background
   - Publishes release with both binaries

## GPU Support

The app automatically detects and uses hardware encoders:

- **NVIDIA** - NVENC (h264_nvenc)
- **AMD** - AMF (h264_amf)
- **Intel** - Quick Sync (h264_qsv)

Enable GPU acceleration in the GUI for faster encoding. Falls back to CPU (libx264) if no GPU is detected.

## Technical Details

### Architecture
- **GUI Framework** - Tkinter (cross-platform)
- **Video Processing** - OpenCV (frame extraction)
- **Encoding** - ffmpeg (batch encoding)
- **Language** - Python 3.10+

### Workflow
1. **Boundary Collection Phase** - Extract middle frame, user selects crop area, store coordinates
2. **Encoding Phase** - Apply stored crop coordinates using ffmpeg filters
3. **Progress Monitoring** - Parse ffmpeg output for real-time progress updates

### ffmpeg Filter Chain
```
crop=W:H:X:Y → scale=TARGET_W:TARGET_H:force_original_aspect_ratio=increase → crop=TARGET_W:TARGET_H
```

## Troubleshooting

**Issue:** App won't open on macOS
**Solution:** Right-click the app and select "Open" to bypass Gatekeeper

**Issue:** No GPU detected
**Solution:** Install GPU-enabled ffmpeg build for your platform

**Issue:** ffmpeg not found
**Solution:** Ensure ffmpeg is installed and in your system PATH

## Contributing

Contributions are welcome! Please:
- Fork the repository
- Create a feature branch
- Add tests for new functionality
- Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

Developed with:
- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [ffmpeg](https://ffmpeg.org/)
- [PyInstaller](https://www.pyinstaller.org/)

---

<div align="center">
  <strong>Enjoy faster, cleaner video crops!</strong>
</div>
