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
import identifier
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


def get_head_yaw(landmarks):
    """Calculates horizontal look ratio. >1.1 is Left, <0.9 is Right, ~1.0 is Center"""
    nose_tip = landmarks['nose_tip'][2]
    jaw_left = landmarks['chin'][0]
    jaw_right = landmarks['chin'][16]
    
    dist_left = nose_tip[0] - jaw_left[0]
    dist_right = jaw_right[0] - nose_tip[0]
    
    return dist_left / (dist_right + 1e-5) # Prevent division by zero

def get_head_pitch(landmarks):
    """Calculates vertical look ratio. >1.5 is Looking Up"""
    nose_tip = landmarks['nose_tip'][2]
    nose_bridge_top = landmarks['nose_bridge'][0]
    jaw_bottom = landmarks['chin'][8]
    
    dist_eyes_to_nose = nose_tip[1] - nose_bridge_top[1]
    dist_nose_to_chin = jaw_bottom[1] - nose_tip[1]
    
    return dist_nose_to_chin / (dist_eyes_to_nose + 1e-5)

def draw_alignment_oval(image, box):
    """Draws a target oval and returns True if the face is correctly inside it"""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # The ideal face box size and position
    top, right, bottom, left = box
    face_center_x = (left + right) // 2
    face_center_y = (top + bottom) // 2
    face_width = right - left
    
    # Check if face is centered (within 30 pixels) and right size (approx 80-130 pixels wide)
    is_aligned = (abs(center_x - face_center_x) < 30) and \
                 (abs(center_y - face_center_y) < 30) and \
                 (80 < face_width < 140)
                 
    color = (0, 255, 0) if is_aligned else (0, 0, 255) # Green if good, Red if bad
    cv2.ellipse(image, (center_x, center_y), (50, 70), 0, 0, 360, color, 2)
    
    return is_aligned


def get_training_base_msg(identifier, uid):
    if not uid or uid == "DENIED_NEW_USER" or uid in getattr(identifier, 'trained', []):
        return None
    if hasattr(identifier, 'training_sessions') and uid in identifier.training_sessions:
        stage = identifier.training_sessions[uid].get('stage', '')
        stages = {
            'CENTER': "TRAINING 1/5: LOOK STRAIGHT AT CAMERA",
            'SLIGHT_LEFT': "TRAINING 2/5: TURN HEAD SLIGHTLY LEFT",
            'FAR_LEFT': "TRAINING 3/5: TURN HEAD FAR LEFT",
            'SLIGHT_RIGHT': "TRAINING 4/5: TURN HEAD SLIGHTLY RIGHT",
            'FAR_RIGHT': "TRAINING 5/5: TURN HEAD FAR RIGHT"
        }
        return stages.get(stage)
    return "TRAINING: WAITING TO START"

async def process_guided_enrollment(identifier, scaled_bgr, face_img_highres_bgr, scaled_rgb, box, landmarks, uid):
    if uid in identifier.trained:
        return True 

    if not hasattr(identifier, 'training_sessions'):
        identifier.training_sessions = {}
        
    if uid not in identifier.training_sessions:
        identifier.training_sessions[uid] = {'stage': 'CENTER', 'last_capture': 0, 'success_msg': ''}
        
    session = identifier.training_sessions[uid]
    base_msg = get_training_base_msg(identifier, uid)
    
    time_since_last = time.time() - session['last_capture']
    if time_since_last < 1.5 and session['last_capture'] != 0:
        # Append success message to the newly advanced stage
        identifier.status_message = f"{base_msg} ({session['success_msg']})"
        return False 

    if check_sharpness(face_img_highres_bgr) < BLUR_THRESHOLD:
        # Append blurry warning to current instruction
        identifier.status_message = f"{base_msg} (Blurry)"
        return False
        
    yaw = get_head_yaw(landmarks)

    print(f"[DEBUG ENROLLMENT] Stage: {session['stage']} | Yaw Ratio: {yaw:.2f} | Target: ", end="")
    if session['stage'] == 'CENTER': print("0.75 to 1.25")
    elif session['stage'] == 'SLIGHT_LEFT': print(">= 1.25")
    elif session['stage'] == 'FAR_LEFT': print(">= 1.55")
    elif session['stage'] == 'SLIGHT_RIGHT': print("<= 0.75")
    elif session['stage'] == 'FAR_RIGHT': print("<= 0.45")

    # -------------------------------------------------------------
    # NEW: On-Screen Visual Guides for Head Turning 
    # -------------------------------------------------------------
    nose_tip = landmarks['nose_tip'][2]
    jaw_left = landmarks['chin'][0]   # Left edge of the face
    jaw_right = landmarks['chin'][16] # Right edge of the face

    # Draw the anchor points (Yellow nose, Pink jaw edges)
    cv2.circle(scaled_bgr, nose_tip, 5, (0, 255, 255), -1) 
    cv2.circle(scaled_bgr, jaw_left, 5, (255, 105, 180), -1) 
    cv2.circle(scaled_bgr, jaw_right, 5, (255, 105, 180), -1)

    # Draw dynamic guide lines and arrows based on the current stage
    if session['stage'] in ['SLIGHT_LEFT', 'FAR_LEFT']:
        # Goal: Maximize the left distance. Draw green active line, red inactive line.
        cv2.line(scaled_bgr, jaw_left, nose_tip, (0, 255, 0), 3)
        cv2.line(scaled_bgr, jaw_right, nose_tip, (0, 0, 255), 1)
        # Arrow showing which way the nose needs to travel on-screen to stretch the green line
        cv2.arrowedLine(scaled_bgr, nose_tip, (nose_tip[0] + 50, nose_tip[1]), (0, 255, 0), 3, tipLength=0.3)

    elif session['stage'] in ['SLIGHT_RIGHT', 'FAR_RIGHT']:
        # Goal: Maximize the right distance. 
        cv2.line(scaled_bgr, jaw_right, nose_tip, (0, 255, 0), 3)
        cv2.line(scaled_bgr, jaw_left, nose_tip, (0, 0, 255), 1)
        # Arrow showing nose travel direction
        cv2.arrowedLine(scaled_bgr, nose_tip, (nose_tip[0] - 50, nose_tip[1]), (0, 255, 0), 3, tipLength=0.3)
        
    elif session['stage'] == 'CENTER':
        # Goal: Balance both lines
        cv2.line(scaled_bgr, jaw_left, nose_tip, (255, 255, 0), 2)
        cv2.line(scaled_bgr, jaw_right, nose_tip, (255, 255, 0), 2)


    # NEW: Async helper function to prevent thread blocking
    async def capture_and_advance(next_stage, success_msg):
        # Offloaded to background thread
        encodings = await asyncio.to_thread(fr.face_encodings, scaled_rgb, [box])
        if encodings:
            identifier.encodings[uid] = np.average([encodings[0], identifier.encodings[uid]], axis=0, weights=[1, 2])
            session['stage'] = next_stage
            session['last_capture'] = time.time()
            session['success_msg'] = success_msg
            
            next_base = get_training_base_msg(identifier, uid)
            identifier.status_message = f"{next_base} ({success_msg})"
            cv2.circle(scaled_bgr, (20, 40), 10, (0, 255, 0), -1)

    if session['stage'] == 'CENTER':
        identifier.status_message = base_msg
        # Widened the center tolerance from ±10% to ±25%
        if 0.75 <= yaw <= 1.25:
            await capture_and_advance('SLIGHT_LEFT', "PERFECT!")
            
    elif session['stage'] == 'SLIGHT_LEFT':
        identifier.status_message = base_msg
        if 1.25 <= yaw <= 1.45:
            await capture_and_advance('FAR_LEFT', "GOT IT!")
            
    elif session['stage'] == 'FAR_LEFT':
        identifier.status_message = base_msg
        if yaw >= 1.60:
            await capture_and_advance('SLIGHT_RIGHT', "EXCELLENT!")
            
    elif session['stage'] == 'SLIGHT_RIGHT':
        identifier.status_message = base_msg
        if 0.55 <= yaw <= 0.75:
            await capture_and_advance('FAR_RIGHT', "ALMOST DONE!")
            
    elif session['stage'] == 'FAR_RIGHT':
        identifier.status_message = base_msg
        if yaw <= 0.40:
            # Offloaded to background thread
            encodings = await asyncio.to_thread(fr.face_encodings, scaled_rgb, [box])
            if encodings:
                identifier.encodings[uid] = np.average([encodings[0], identifier.encodings[uid]], axis=0, weights=[1, 2])
                identifier.trained.append(uid) 
                identifier.saveMeta() 
                
                identifier.status_message = "TRAINING COMPLETE! (Saving...)"
                session['last_capture'] = time.time()
                cv2.circle(scaled_bgr, (20, 40), 10, (0, 255, 0), -1)
                
    return False

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

    # --- OPTIMIZATION VARIABLES ---
    frame_count = 0
    process_this_frame_interval = 4  # Only run face detection 1 out of every 4 frames
    face_locations = []


    while True:
        await asyncio.sleep(0.005)  # Keep loop responsive
        if identifier.exit:
            break

        try:
            # capture_array() returns a numpy array in the configured format (BGR888)
            frame_raw_rgb = picam2.capture_array()
        except Exception as e:
            print(f"Camera capture error: {e}")
            continue


        # 1. THE FIX: Immediately translate the raw RGB frame into OpenCV's native BGR format
        frame_bgr = frame_raw_rgb

        # 2. Scale down the BGR frame for drawing and streaming
        scaled_bgr = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5)

        # 3. Create a clean RGB copy just for the AI to use
        scaled_rgb = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB)

        # --- OPTIMIZATION 1: FRAME SKIPPING ---
        if frame_count % process_this_frame_interval == 0:
            # --- OPTIMIZATION 2: OFFLOAD CPU TO THREAD ---
            # This prevents the AI from freezing your Sanic dashboard stream
            # Pass the RGB copy to the AI to find the faces
            face_locations = await asyncio.to_thread(fr.face_locations, scaled_rgb)
        
        frame_count += 1

        # MULTIPLE FACE DETECTION
        if len(face_locations) > 1:
            session_timeout_frames = 0
            # Append multiple face error if training, otherwise show standard warning
            base_msg = get_training_base_msg(identifier, current_session_person)
            if base_msg:
                identifier.status_message = f"{base_msg} (Multiple faces detected)"
            else:
                identifier.status_message = "WARNING: MULTIPLE FACES DETECTED. Please keep only ONE face in frame."
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                cv2.rectangle(scaled_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
                # Keep face numbers on screen for debugging
                cv2.putText(scaled_bgr, f"Face {i+1}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # EXACTLY ONE FACE DETECTED
        elif len(face_locations) == 1:
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
                    identifier.status_message = "MOTION BLUR: PLEASE HOLD STILL"
                else:
                    identifier.status_message = "SCANNING FACE..."
                    
                    try:
                        # Only run the heavy ResNet encoder IF the image is sharp
                        encodings = await asyncio.to_thread(fr.face_encodings, scaled_rgb, [(top, right, bottom, left)])

                        if len(encodings) > 0:
                            face_encoding = encodings[0]
                            person = identifier.getIDFromEncoding(face_encoding)

                            if person is None:
                                # Because it passed the sharpness test, this saved image will be perfectly clear!
                                
                                # 1. Save the cropped face to the database and get the new user's ID
                                new_uid = identifier.addNew(face_img_highres_bgr, face_encoding)
                                
                                # 2. Save the FULL uncropped security frame using that same ID
                                cv2.imwrite(f'peoplefullframe/{new_uid}_full_frame.jpg', frame_bgr)
                                
                                accessDenied(name="New User", reason="Not Enrolled")
                                identifier.status_message = "NEW USER CAPTURED - ACCESS DENIED"
                                
                                # Lock them out of this session so it doesn't span multiple files
                                current_session_person = "DENIED_NEW_USER"
                            else:
                                current_session_person = person
                                
                                if not identifier.hasAccess(current_session_person):
                                    person_name = identifier.friendly_names.get(person, person)
                                    accessDenied(name=person_name, reason="Policy Denied")
                                    identifier.status_message = f"ACCESS DENIED: {person_name} is not authorized."

                    except IndexError:
                        pass
            
            # -------------------------------------------------------------------
            # STAGE 2: LIGHTWEIGHT LIVENESS & GUIDED ENROLLMENT
            # -------------------------------------------------------------------
            else:
                if current_session_person != "DENIED_NEW_USER" and identifier.hasAccess(current_session_person):
                    
                    face_landmarks_list = fr.face_landmarks(scaled_rgb, [(top, right, bottom, left)])
                    box = (top, right, bottom, left)

                    # ALWAYS draw the alignment oval so they know where to stand
                    is_aligned = draw_alignment_oval(scaled_bgr, box)

                    if face_landmarks_list:
                        landmarks = face_landmarks_list[0]
                        
                        # 1. RUN THE ENROLLMENT CHECK
                        is_trained = await process_guided_enrollment(
                                        identifier, scaled_bgr, face_img_highres_bgr, scaled_rgb, box, landmarks, current_session_person
                                    )
                                            
                        # 2. ONLY ALLOW BLINK UNLOCK IF FULLY TRAINED
                        if is_trained:
                            if not is_aligned:
                                identifier.status_message = "ALIGNMENT: PLEASE PLACE FACE IN THE OVAL"
                                if hasattr(identifier, 'liveness_session'):
                                    del identifier.liveness_session # Reset if they move away
                            else:
                                # Initialize Randomized Challenge
                                if not hasattr(identifier, 'liveness_session'):
                                    import random
                                    challenges = ['LOOK_UP', 'BLINK', 'LOOK_LEFT', 'LOOK_RIGHT']
                                    identifier.liveness_session = {
                                        'challenge': random.choice(challenges),
                                        'frames_held': 0
                                    }
                                
                                session = identifier.liveness_session
                                passed_frame = False
                                
                                # Evaluate the specific random challenge
                                if session['challenge'] == 'LOOK_UP':
                                    identifier.status_message = "SECURITY CHECK: LOOK SLIGHTLY UP"
                                    pitch = get_head_pitch(landmarks)
                                    if pitch > 1.55: # Threshold for looking up
                                        passed_frame = True
                                        cv2.putText(scaled_bgr, "HOLD IT...", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                                elif session['challenge'] == 'BLINK':
                                    identifier.status_message = "SECURITY CHECK: BLINK YOUR EYES"
                                    left_eye = landmarks.get('left_eye')
                                    right_eye = landmarks.get('right_eye')
                                    if left_eye and right_eye:
                                        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                                        if ear < EYE_AR_THRESH:
                                            passed_frame = True
                                            cv2.putText(scaled_bgr, "BLINK DETECTED...", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                                elif session['challenge'] == 'LOOK_LEFT':
                                    identifier.status_message = "SECURITY CHECK: TURN HEAD SLIGHTLY LEFT"
                                    yaw = get_head_yaw(landmarks)
                                    if yaw > 1.25:
                                        passed_frame = True
                                        cv2.putText(scaled_bgr, "HOLD IT...", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                        
                                elif session['challenge'] == 'LOOK_RIGHT':
                                    identifier.status_message = "SECURITY CHECK: TURN HEAD SLIGHTLY RIGHT"
                                    yaw = get_head_yaw(landmarks)
                                    if yaw < 0.75:
                                        passed_frame = True
                                        cv2.putText(scaled_bgr, "HOLD IT...", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                                # Require the user to hold the action for a few frames (resolves low FPS issues)
                                if passed_frame:
                                    session['frames_held'] += 1
                                else:
                                    # Don't reset the counter for blinking, since eyes naturally open quickly
                                    if session['challenge'] != 'BLINK':
                                        session['frames_held'] = 0

                                # Challenge Completed!
                                if session['frames_held'] >= 3:
                                    person_name = identifier.friendly_names.get(current_session_person, current_session_person)
                                    accessGranted(name=person_name)
                                    
                                    identifier.status_message = f"UNLOCKED: WELCOME {person_name.upper()}!"
                                    ret_enc, v = cv2.imencode('.jpg', scaled_bgr)
                                    identifier.setView(v.tobytes())
                                    
                                    await asyncio.sleep(3) 
                                    current_session_person = None 
                                    del identifier.liveness_session # Reset challenge for next person
                else:
                    person_name = identifier.friendly_names.get(current_session_person, current_session_person)
                    identifier.status_message = f"ACCESS DENIED: {person_name} is not authorized."
        
        # NO FACES DETECTED
        else:
            # If face disappears for a few frames, reset the state machine
            session_timeout_frames += 1
            # Append no-face error if training, otherwise show standard idle message
            base_msg = get_training_base_msg(identifier, current_session_person)
            if base_msg:
                identifier.status_message = f"{base_msg} (No face detected)"
            else:
                identifier.status_message = "System Idle - Waiting for face..."
                
            if session_timeout_frames > 10:
                current_session_person = None
                blink_counter = 0

        # Encode frame and stream to dashboard
        ret_enc, v = cv2.imencode('.jpg', scaled_bgr)
        if ret_enc:
            identifier.setView(v.tobytes())

    picam2.stop()