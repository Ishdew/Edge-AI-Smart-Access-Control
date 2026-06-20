# Edge-AI-Smart-Access-Control

An advanced edge AI facial recognition door lock system featuring a web dashboard, guided enrollment, anti-spoofing liveness detection, and Raspberry Pi OS Trixie compatibility.

This project is an evolution of a basic facial recognition door lock, introducing robust security features to ensure only authorized individuals—and not 2D photographs or spoof attempts—gain access.

## ✨ Key Features

- **Anti-Spoofing & Liveness Detection**: Employs blink detection using Eye Aspect Ratio (EAR) and randomized challenge-response (look up, turn head left/right, blink) to ensure a live 3D face is present.
- **Guided User Enrollment**: Interactively guides new users through an enrollment phase, capturing multiple head angles and running blurriness checks to guarantee high-quality dataset entries.
- **Web Dashboard**: A live-updating web interface built with Sanic and Vue.js. Provides a live camera feed alongside user management (grant/deny access, rename, delete) right from your browser.
- **Audit Logging**: Keeps an access trail in `access_audit.log`, noting who was granted or denied access and the reason why.
- **Raspberry Pi OS Trixie Support**: Fully upgraded to the libcamera stack (`picamera2`) and modern GPIO backends (`gpiozero` with `lgpio`) for compatibility with the latest Raspberry Pi OS releases.
- **Laptop Simulator Mode**: Automatically falls back to a simulated mode (using the laptop webcam and skipping GPIO commands) if a Pi camera or GPIO pins aren't detected.

## ⚙️ Hardware Requirements

- **Raspberry Pi**: Pi 3 or Pi 4 recommended.
- **Camera Module**: Raspberry Pi Camera Module (compatible with libcamera).
- **Relay Module**: Connected to **GPIO 14** to control a physical electronic door strike or magnetic lock.

## 🛠️ Software Requirements

- Python 3
- OpenCV (`cv2`)
- Sanic (Async Web Framework)
- `face_recognition` (dlib-based)
- `numpy`, `asyncio`, `json`
- `picamera2` (for PiOS Trixie)
- `gpiozero`

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Edge-AI-Smart-Access-Control.git
   cd Edge-AI-Smart-Access-Control
   ```

2. **Run the setup script:**
   The `setup.sh` script automates the installation of system packages, Python libraries, increases the swap size (needed for compiling dlib/face_recognition), and installs the `doorlock` service.
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Reboot the system:**
   ```bash
   sudo reboot
   ```

## 🎮 Usage

Once the setup is complete, the `doorlock` service will run automatically on boot. 

- **Start/Stop the Service manually:**
  ```bash
  sudo /etc/init.d/doorlock start
  sudo /etc/init.d/doorlock stop
  ```

- **Access the Web Dashboard:**
  Open a browser on any device connected to the same network and navigate to the Raspberry Pi's IP address on port 8080:
  ```
  http://<raspberry_pi_ip>:8080/
  ```

  From the dashboard, you can:
  - View the live camera feed and real-time status prompts.
  - See unfamiliar faces as they are captured (denied by default).
  - Assign friendly names to users.
  - Toggle their access rights (`Access Granted` / `Access Denied`).
  - Delete user records entirely.

## 📁 Project Structure

- `src/doorlock.py`: Main Sanic application server. Handles routing and API endpoints.
- `src/identifier.py`: Core class for managing face encodings, user permissions, and persistent metadata.
- `src/functions_trixie.py`: Video processing loop utilizing `picamera2`. Contains the logic for face detection, liveness checks, and guided enrollment.
- `src/functions.py`: Legacy version using `picamera` or a mock webcam stream.
- `src/index/`: HTML/CSS/JS for the Vue.js web dashboard.
- `people/`: Storage directory for saved user face thumbnails and `meta.txt` (permissions and names).
- `access_audit.log`: Generated runtime log tracking all access attempts.