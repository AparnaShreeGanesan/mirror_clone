# MirrorCloneFX

irrorCloneFX

MirrorCloneFX is a real-time computer vision application that creates stylized visual clones of yourself using a webcam. It uses hand gesture recognition powered by MediaPipe and OpenCV to switch between dynamic visual effects instantly.

Features

Split-screen view
Displays the original webcam feed alongside a stylized clone in real time.

Gesture-based controls
Switch effects using simple hand gestures detected via MediaPipe hand tracking.

Multiple visual styles

Dots – Stippled dot rendering with variable sizes

Lines – Edge-based outline rendering with enhanced color effects

ASCII – Unicode-based text art representation

Particles – Physics-driven particles that follow movement dynamically

Hand Gesture Controls
Gesture	Effect	Description
Two fingers (peace sign)	Dots	Creates a stippled dot version using original colors
One finger up	Lines	Shows edge outlines and contours with glow effects
Thumb + pinky out	ASCII	Renders the image as ASCII art with color intensity mapping
Open palm (all fingers)	Particles	Generates physics-based particles that follow movement
Installation
Prerequisites

Python 3.7 or higher

A working webcam

Good lighting for accurate hand detection

Setup

Clone the repository

git clone https://github.com/tubakhxn/MirrorCloneFX.git
cd MirrorCloneFX


Install dependencies

pip install -r requirements.txt


Run the application

python main.py

Usage

Position yourself clearly in front of the webcam

Keep your hand visible in the camera frame

Perform hand gestures to switch effects

Press q to exit the application

The currently active mode is displayed in the top-left corner of the window.

Technical Overview
Core Technologies

OpenCV – Video capture and image processing

MediaPipe – Real-time hand landmark detection

NumPy – Efficient numerical operations

Visual Effects

Dots – Intensity-based circular rendering

Lines – Canny edge detection with color enhancement

ASCII – Unicode block character mapping

Particles – Physics-based system with gravity and momentum

Performance

Runs at ~30 FPS on modern hardware

Tracks 21 hand landmarks in real time

Optimized for smooth transitions between effects

System Requirements
Hardware

OS: Windows, macOS, or Linux

RAM: 4GB minimum (8GB recommended)

CPU: Multi-core processor recommended

Webcam: USB or built-in camera

Software Dependencies

opencv-python >= 4.5.0

mediapipe >= 0.8.9

numpy >= 1.21.0

Contributing

Contributions are welcome and appreciated.

How to Contribute

Report bugs via GitHub Issues

Suggest new effects or improvements

Submit pull requests for features or fixes

Improve documentation or testing coverage

Development Workflow

Fork the repository

Clone your fork

git clone https://github.com/YOUR-USERNAME/MirrorCloneFX.git


Create a feature branch

git checkout -b feature-name


Make and test your changes

Commit and push

Open a pull request

Code Guidelines

Follow PEP 8

Add docstrings and comments

Ensure compatibility with Python 3.7+

Adding New Visual Effects

Create a new method in the main class:

def create_custom_effect(self, frame):
    return processed_frame


Register the effect in the modes dictionary

Assign a gesture trigger

Test under different lighting conditions

Update documentation accordingly

Troubleshooting
Common Issues

Webcam not detected

Ensure it’s not in use by another app

Check system camera permissions

Gesture recognition issues

Improve lighting

Keep hand fully visible

Hold gestures steady for 1–2 seconds

Performance drops

Close background applications

Improve lighting to reduce processing load

Dependency installation errors

Update pip

Install packages individually if needed

Platform Notes

Windows

May require Visual C++ redistributables

macOS

Camera permission may need manual approval

Linux

Additional OpenCV packages may be required

Customization

Each visual effect can be tweaked by adjusting parameters such as:

Density and size (Dots)

Edge thresholds and glow intensity (Lines)

Character sets and spacing (ASCII)

Physics parameters and particle count (Particles)

License

Released under the MIT License. See the LICENSE file for details.

Acknowledgments

MediaPipe team for real-time hand tracking

OpenCV community for computer vision tools

Python open-source ecosystem

Connect

GitHub: https://github.com/tubakhxn

