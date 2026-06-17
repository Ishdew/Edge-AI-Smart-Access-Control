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

def get_head_orientation(landmarks):
    """Calculates Yaw and Pitch using 2D facial landmark geometry"""
    
    # Grab the specific x,y coordinates we need
    nose_tip = landmarks['nose_tip'][2]
    jaw_left = landmarks['chin'][0]
    jaw_right = landmarks['chin'][16]
    jaw_bottom = landmarks['chin'][8]
    nose_bridge_top = landmarks['nose_bridge'][0]

    # Calculate horizontal distances (Yaw)
    dist_nose_to_left_jaw = nose_tip[0] - jaw_left[0]
    dist_nose_to_right_jaw = jaw_right[0] - nose_tip[0]

    # Calculate vertical distances (Pitch)
    dist_nose_to_eyes = nose_tip[1] - nose_bridge_top[1]
    dist_nose_to_chin = jaw_bottom[1] - nose_tip[1]

    # Evaluate ratios to determine orientation
    if dist_nose_to_left_jaw > dist_nose_to_right_jaw * 1.8:
        return "LOOKING_RIGHT"
    elif dist_nose_to_right_jaw > dist_nose_to_left_jaw * 1.8:
        return "LOOKING_LEFT"
    elif dist_nose_to_chin > dist_nose_to_eyes * 1.5:
        return "LOOKING_UP"
    elif dist_nose_to_chin < dist_nose_to_eyes * 0.8:
        return "LOOKING_DOWN"
    else:
        return "CENTER"

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
            # capture_array() returns a numpy array in the configured format (BGR888)
            frame_raw_rgb = picam2.capture_array()
        except Exception as e:
            print(f"Camera capture error: {e}")
            continue


        # 1. THE FIX: Immediately translate the raw RGB frame into OpenCV's native BGR format
        frame_bgr = cv2.cvtColor(frame_raw_rgb, cv2.COLOR_RGB2BGR)

        # 2. Scale down the BGR frame for drawing and streaming
        scaled_bgr = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5)

        # 3. Create a clean RGB copy just for the AI to use
        scaled_rgb = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB)

        # Pass the RGB copy to the AI to find the faces
        face_locations = fr.face_locations(scaled_rgb)

        if len(face_locations) > 0:
            session_timeout_frames = 0 # Face is present, reset timeout
            top, right, bottom, left = face_locations[0]

            # Draw a rectangle on the BGR frame for this face
            cv2.rectangle(scaled_bgr, (left, top), (right, bottom), (255, 0, 0), 2)

            # High-res crop for saving/blur checking
            orig_top = max(0, top * 2)
            orig_bottom = min(frame_bgr.shape[0], bottom * 2)
            orig_left = max(0, left * 2)
            orig_right = min(frame_bgr.shape[1], right * 2)
            face_img_highres_bgr = frame_bgr[orig_top:orig_bottom, orig_left:orig_right]

            # -------------------------------------------------------------------
            # STAGE 1: ENCODE ONCE & BLUR CHECK GATEKEEPER
            # -------------------------------------------------------------------
            if current_session_person is None:
                # 1A. Check if the user is holding still
                sharpness = check_sharpness(face_img_highres_bgr)
                
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
                            
                            # 1. Save the cropped face to the database and get the new user's ID
                            new_uid = identifier.addNew(face_img_highres_bgr, face_encoding)
                            
                            # 2. Save the FULL uncropped security frame using that same ID
                            cv2.imwrite(f'people/{new_uid}_full_frame.jpg', frame_bgr)
                            
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
            # STAGE 2: LIGHTWEIGHT LIVENESS & GUIDED ENROLLMENT
            # -------------------------------------------------------------------
            else:
                if current_session_person != "DENIED_NEW_USER" and identifier.hasAccess(current_session_person):
                    
                    # Ensure we have a tracking variable for their enrollment progress
                    if not hasattr(identifier, 'enrollment_stage'):
                        identifier.enrollment_stage = "NEED_LEFT"

                    # Get landmarks for both blink detection and pose estimation
                    face_landmarks_list = fr.face_landmarks(scaled_rgb, [(top, right, bottom, left)])
                    
                    if face_landmarks_list:
                        landmarks = face_landmarks_list[0]
                        
                        # 1. Check head orientation
                        current_pose = get_head_orientation(landmarks)


                        # --- HELPER FUNCTION FOR TOP-CENTERED TEXT ---
                        def draw_centered_instruction(text, color):
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            scale = 0.7
                            thickness = 2
                            # Get the width and height of the text box
                            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
                            # Calculate the perfect X center: (Frame Width - Text Width) / 2
                            text_x = (scaled_bgr.shape[1] - text_size[0]) // 2
                            text_y = 30 # 30 pixels down from the very top of the frame
                            
                            # Draw a subtle black outline for better readability against bright backgrounds
                            cv2.putText(scaled_bgr, text, (text_x, text_y), font, scale, (0, 0, 0), thickness + 2)
                            # Draw the main colored text
                            cv2.putText(scaled_bgr, text, (text_x, text_y), font, scale, color, thickness)
                        # ---------------------------------------------

                        # 2. Guide the user through the training stages
                        if identifier.enrollment_stage == "NEED_LEFT":
                            draw_centered_instruction("TRAINING: TURN HEAD LEFT", (0, 165, 255))
                            
                            if current_pose == "LOOKING_LEFT":
                                identifier.enrollment_stage = "NEED_RIGHT"
                                cv2.putText(scaled_bgr, "GOOD!", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                        elif identifier.enrollment_stage == "NEED_RIGHT":
                            draw_centered_instruction("TRAINING: TURN HEAD RIGHT", (0, 165, 255))
                            
                            if current_pose == "LOOKING_RIGHT":
                                identifier.enrollment_stage = "NEED_BLINK"
                                cv2.putText(scaled_bgr, "GOOD!", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                        elif identifier.enrollment_stage == "NEED_BLINK":
                            draw_centered_instruction("TRAINING COMPLETE: PLEASE BLINK", (0, 0, 255))
                            
                            left_eye = landmarks.get('left_eye')
                            right_eye = landmarks.get('right_eye')
                            if left_eye and right_eye:
                                leftEAR = calculate_ear(left_eye)
                                rightEAR = calculate_ear(right_eye)
                                ear = (leftEAR + rightEAR) / 2.0

                                if ear < EYE_AR_THRESH:
                                    blink_counter += 1
                                else:
                                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                                        person_name = identifier.friendly_names.get(current_session_person, current_session_person)
                                        accessGranted(name=person_name)
                                        
                                        # Center the unlock success message as well
                                        draw_centered_instruction(f"UNLOCKED: WELCOME {person_name.upper()}", (0, 255, 0))
                                        
                                        ret_enc, v = cv2.imencode('.jpg', scaled_bgr)
                                        identifier.setView(v.tobytes())
                                        
                                        await asyncio.sleep(3) 
                                        current_session_person = None 
                                        identifier.enrollment_stage = "NEED_LEFT" 
                                        
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