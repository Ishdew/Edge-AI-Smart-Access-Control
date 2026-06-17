"""
functions_trixie.py — Raspberry Pi OS Trixie compatible version

Changes from the original Pi version:
  - Replaces legacy `picamera` with `picamera2` (libcamera stack)
  - Uses `gpiozero` with `lgpio` backend (RPi.GPIO is deprecated on Trixie)
  - All face recognition, blink detection, and streaming logic is identical
"""

import cv2
import numpy as np
import asyncio
import face_recognition as fr
import time
import gpiozero
import logging

# --- Picamera2 (libcamera) ---
from picamera2 import Picamera2

# --- Setup Audit Logging ---
logging.basicConfig(filename='access_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Lazy Relay Initialization ---
# GPIO 14 is the default UART TX pin. If the serial console is enabled,
# it will be "busy". We lazy-init to avoid crashing at import time
# (which also prevents Sanic's multiprocess spawn from failing).
_relay = None
_relay_initialized = False

def _get_relay():
    global _relay, _relay_initialized
    if not _relay_initialized:
        _relay_initialized = True
        try:
            _relay = gpiozero.LED(14)
            print("[GPIO] Relay initialized on GPIO 14")
        except Exception as e:
            print(f"[GPIO] Could not initialize relay on GPIO 14: {e}")
            print("[GPIO] Running in SIMULATOR mode — no physical door lock control")
            _relay = None
    return _relay

# --- Blink Detection Constants ---
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 2
BLUR_THRESHOLD = 80.0  # Increase to 100 for stricter sharpness, decrease to 50 for less strict

def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR) for liveness detection"""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0

def check_sharpness(image_bgr):
    """Calculates the focus measure of an image using Laplacian Variance"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def accessGranted(name=None):
    relay = _get_relay()
    if relay:
        relay.blink(5, 1, 1)  # Unlock door for 5 seconds
    log_msg = f"Access Granted for user: {name}" if name else "Access Granted: Unknown"
    print(f"\n[DOOR] UNLOCKED! - {log_msg}\n")
    logging.info(log_msg)

def accessDenied(name=None, reason="Unauthorized"):
    relay = _get_relay()
    if relay:
        relay.off()
    log_msg = f"Access Denied for user: {name}. Reason: {reason}" if name else f"Access Denied. Reason: {reason}"
    print(f"[DOOR] LOCKED - {log_msg}")
    logging.info(log_msg)

# ---------------------------------------------------------------------------------------

async def videoProcessing(identifier, imshow=False):
    # Initialize Picamera2 (libcamera stack — works on Trixie)
    picam2 = Picamera2()

    # Configure for video capture: RGB888 format gives us a numpy array directly
    # 640x480 is a good balance of quality vs. processing speed on Pi 4
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    print('Started Picamera2 video stream (libcamera)')
    await asyncio.sleep(0.5)  # Allow camera to warm up

    # --- STATE MACHINE VARIABLES ---
    blink_counter = 0
    current_session_person = None  # Caches the recognized person
    session_timeout_frames = 0     # Resets session if face disappears

    while True:
        await asyncio.sleep(0.01)  # Keep loop responsive
        if identifier.exit:
            break

        try:
            # capture_array() returns a numpy array in the configured format (RGB888)
            frame = picam2.capture_array()
        except Exception as e:
            print(f"Camera capture error: {e}")
            continue

        # Picamera2 with RGB888 gives us RGB directly, but OpenCV uses BGR for display
        # face_recognition needs RGB, so we keep the original for FR and convert for drawing
        rgb_frame = frame  # Already RGB from picamera2

        # Scale down for faster face detection processing
        scaled_rgb = cv2.resize(rgb_frame, None, fx=0.5, fy=0.5)

        # Convert to BGR for OpenCV drawing operations
        scaled_bgr = cv2.cvtColor(scaled_rgb, cv2.COLOR_RGB2BGR)

        face_locations = fr.face_locations(scaled_rgb)

        if len(face_locations) > 0:
            session_timeout_frames = 0 # Face is present, reset timeout
            top, right, bottom, left = face_locations[0]

            # Draw a rectangle on the BGR frame for this face
            cv2.rectangle(scaled_bgr, (left, top), (right, bottom), (255, 0, 0), 2)

            # High-res crop for saving/blur checking
            orig_top = max(0, top * 2)
            orig_bottom = min(frame.shape[0], bottom * 2)
            orig_left = max(0, left * 2)
            orig_right = min(frame.shape[1], right * 2)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_img_highres = frame_bgr[orig_top:orig_bottom, orig_left:orig_right]

            # -------------------------------------------------------------------
            # STAGE 1: ENCODE ONCE & BLUR CHECK GATEKEEPER
            # -------------------------------------------------------------------
            if current_session_person is None:
                # 1A. Check if the user is holding still
                sharpness = check_sharpness(face_img_highres)
                
                if sharpness < BLUR_THRESHOLD:
                    cv2.putText(scaled_bgr, "MOTION BLUR: HOLD STILL", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    cv2.putText(scaled_bgr, "SCANNING FACE...", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    try:
                        # Only run the heavy ResNet encoder IF the image is sharp
                        face_encoding = fr.face_encodings(scaled_rgb, [(top, right, bottom, left)])[0]
                        person = identifier.getIDFromEncoding(face_encoding)

                        if person is None:
                            # Because it passed the sharpness test, this saved image will be perfectly clear!
                            identifier.addNew(face_img_highres, face_encoding)
                            accessDenied(name="New User", reason="Not Enrolled")
                            cv2.putText(scaled_bgr, "NEW USER SAVED", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            
                            # Lock them out of this session so it doesn't span multiple files
                            current_session_person = "DENIED_NEW_USER" 
                        else:
                            current_session_person = person
                            
                            person_name = person
                            if person in identifier.friendly_names:
                                person_name = identifier.friendly_names[person]
                                
                            if not identifier.hasAccess(current_session_person):
                                accessDenied(name=person_name, reason="Policy Denied")

                    except IndexError:
                        pass
            
            # -------------------------------------------------------------------
            # STAGE 2: LIGHTWEIGHT LIVENESS (ResNet Bypassed)
            # -------------------------------------------------------------------
            else:
                # If they are a known user with access, run the blink challenge
                if current_session_person != "DENIED_NEW_USER" and identifier.hasAccess(current_session_person):
                    
                    cv2.putText(scaled_bgr, "AUTHORIZED: PLEASE BLINK", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    face_landmarks_list = fr.face_landmarks(scaled_rgb, [(top, right, bottom, left)])
                    if face_landmarks_list:
                        landmarks = face_landmarks_list[0]
                        left_eye = landmarks.get('left_eye')
                        right_eye = landmarks.get('right_eye')

                        if left_eye and right_eye:
                            for point in left_eye + right_eye:
                                cv2.circle(scaled_bgr, point, 2, (0, 255, 255), -1)

                            leftEAR = calculate_ear(left_eye)
                            rightEAR = calculate_ear(right_eye)
                            ear = (leftEAR + rightEAR) / 2.0

                            cv2.putText(scaled_bgr, f"Live EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                            if ear < EYE_AR_THRESH:
                                blink_counter += 1
                                cv2.putText(scaled_bgr, "BLINKING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            else:
                                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                                    # Liveness passed! 
                                    person_name = identifier.friendly_names.get(current_session_person, current_session_person)
                                    accessGranted(name=person_name)
                                    
                                    cv2.putText(scaled_bgr, "UNLOCKED", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    ret_enc, v = cv2.imencode('.jpg', scaled_bgr)
                                    identifier.setView(v.tobytes())
                                    
                                    await asyncio.sleep(3) # Hold door open
                                    current_session_person = None # Reset for next person
                                    
                                blink_counter = 0
                else:
                    cv2.putText(scaled_bgr, "ACCESS DENIED", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            # If face disappears for a few frames, reset the state machine
            session_timeout_frames += 1
            if session_timeout_frames > 10:
                current_session_person = None
                blink_counter = 0

        # Encode frame and stream to dashboard
        ret_enc, v = cv2.imencode('.jpg', scaled_bgr)
        if ret_enc:
            identifier.setView(v.tobytes())

    picam2.stop()