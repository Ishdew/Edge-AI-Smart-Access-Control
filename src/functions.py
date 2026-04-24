# import cv2
# import numpy as np
# import asyncio
# import face_recognition as fr
# import time
# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import gpiozero
# import logging

# # --- Setup Audit Logging ---
# logging.basicConfig(filename='access_audit.log', level=logging.INFO,
#                     format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# # Initialize the Relay
# relay = gpiozero.LED(14)

# # --- Blink Detection Constants ---
# EYE_AR_THRESH = 0.20
# EYE_AR_CONSEC_FRAMES = 2

# def calculate_ear(eye):
#     """Calculate the Eye Aspect Ratio (EAR) for liveness detection"""
#     A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
#     B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
#     C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
#     ear = (A + B) / (2.0 * C)
#     return ear

# def accessGranted(name=None):
#     relay.blink(5, 1, 1) # Unlock door for 5 seconds
#     log_msg = f"Access Granted for user: {name}" if name else "Access Granted: Unknown"
#     print(log_msg)
#     logging.info(log_msg)

# def accessDenied(name=None, reason="Unauthorized"):
#     relay.off()
#     log_msg = f"Access Denied for user: {name}. Reason: {reason}" if name else f"Access Denied. Reason: {reason}"
#     print(log_msg)
#     logging.info(log_msg)

# # ---------------------------------------------------------------------------------------

# async def videoProcessing(identifier, imshow=False):
#     picam = PiCamera()
#     # Lower resolution is CRITICAL for Pi 3 to maintain low latency/FPS
#     picam.resolution = (640, 480) 
#     raw = PiRGBArray(picam)

#     print('Started video stream')
#     await asyncio.sleep(0.1)

#     blink_counter = 0
#     liveness_passed = False

#     while True:
#         await asyncio.sleep(0.1)
#         if identifier.exit:
#             break
    
#         try:
#             raw.seek(0)
#             raw.truncate()
#             picam.capture(raw, format='bgr', use_video_port=True)
#             frame = raw.array
#         except Exception as e:
#             print(e)
#             continue

#         # Scale down for faster processing
#         scaled = cv2.resize(frame, None, fx=0.5, fy=0.5)
#         # face_recognition requires RGB format
#         rgb_scaled = scaled[:, :, ::-1] 

#         face_locations = fr.face_locations(rgb_scaled)
        
#         if len(face_locations) > 0:
#             # We process the first detected face in the frame
#             top, right, bottom, left = face_locations[0]
            
#             # Draw a rectangle on the frame for this face
#             cv2.rectangle(scaled, (left, top), (right, bottom), (255,0,0), 2)

#             # 1. EXTRACT LANDMARKS FOR LIVENESS CHECK
#             face_landmarks_list = fr.face_landmarks(rgb_scaled, [face_locations[0]])
#             landmarks = face_landmarks_list[0]
            
#             left_eye = landmarks.get('left_eye')
#             right_eye = landmarks.get('right_eye')
            
#             if left_eye and right_eye:
#                 leftEAR = calculate_ear(left_eye)
#                 rightEAR = calculate_ear(right_eye)
#                 ear = (leftEAR + rightEAR) / 2.0

#                 # Check if eyes are closed
#                 if ear < EYE_AR_THRESH:
#                     blink_counter += 1
#                 else:
#                     # If eyes were closed for enough frames, register a blink
#                     if blink_counter >= EYE_AR_CONSEC_FRAMES:
#                         liveness_passed = True
#                         print("Blink detected! Liveness confirmed.")
#                     blink_counter = 0

#             # 2. EXTRACT ENCODING FOR IDENTITY CHECK
#             top *= 2
#             bottom *= 2
#             right *= 2
#             left *= 2
#             face_img = frame[top:bottom, left:right] 

#             try:
#                 await asyncio.sleep(0.1)	
#                 face_encoding = fr.face_encodings(face_img)[0]
#             except Exception as e:
#                 continue

#             person = identifier.getIDFromEncoding(face_encoding)
            
#             # Fetch friendly name for logging, fallback to UID
#             person_name = person
#             if person and person in identifier.friendly_names:
#                 person_name = identifier.friendly_names[person]

#             # 3. DECISION MATRIX (Identity + Policy + Liveness)
#             if person is None:
#                 print('Adding new person')
#                 identifier.addNew(face_img, face_encoding)
#                 accessDenied(name="New User", reason="Not Enrolled")
#                 liveness_passed = False 
            
#             elif identifier.hasAccess(person):
#                 if liveness_passed:
#                     accessGranted(name=person_name)
#                     liveness_passed = False # Reset state after entry
#                     await asyncio.sleep(5)  # Pause processing to allow entry
#                 else:
#                     print(f"User {person_name} identified. Waiting for blink challenge...")
#                     # Prompt user on the stream
#                     cv2.putText(scaled, "PLEASE BLINK", (left//2, top//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             else:
#                 accessDenied(name=person_name, reason="Policy Denied")
#                 liveness_passed = False 
#         else:
#             # Reset state if the person walks away
#             liveness_passed = False
#             blink_counter = 0

#         # Encode and stream the view to the web dashboard
#         ret, v = cv2.imencode('.jpg', scaled)
#         identifier.setView(v)

#     picam.close()
#     cv2.destroyAllWindows()

import cv2
import numpy as np
import asyncio
import face_recognition as fr
import time
import logging

# --- Setup Audit Logging ---
logging.basicConfig(filename='access_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Mocking the Relay for Laptop Testing
# import gpiozero
# relay = gpiozero.LED(14)

# --- Blink Detection Constants ---
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 2

def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR) for liveness detection"""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0

def accessGranted(name=None):
    log_msg = f"Access Granted for user: {name}" if name else "Access Granted: Unknown"
    print(f"\n[SIMULATOR] DOOR UNLOCKED! - {log_msg}\n")
    logging.info(log_msg)

def accessDenied(name=None, reason="Unauthorized"):
    log_msg = f"Access Denied for user: {name}. Reason: {reason}" if name else f"Access Denied. Reason: {reason}"
    print(f"[SIMULATOR] DOOR LOCKED - {log_msg}")
    logging.info(log_msg)

# ---------------------------------------------------------------------------------------

async def videoProcessing(identifier, imshow=False):
    # Initialize Laptop Webcam
    vstream = cv2.VideoCapture(0)
    print('Started laptop video stream')
    await asyncio.sleep(0.1)

    blink_counter = 0
    liveness_passed = False

    while True:
        await asyncio.sleep(0.01) # Keep loop fast to empty camera buffer
        if identifier.exit:
            break
    
        ret, frame = vstream.read()
        if not ret:
            continue

        # Scale down for faster processing
        scaled = cv2.resize(frame, None, fx=0.5, fy=0.5)
        # face_recognition requires RGB format
        rgb_scaled = scaled[:, :, ::-1] 

        face_locations = fr.face_locations(rgb_scaled)
        
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            
            # Draw a rectangle on the frame for this face
            cv2.rectangle(scaled, (left, top), (right, bottom), (255,0,0), 2)

            # 1. EXTRACT LANDMARKS FOR LIVENESS CHECK
            face_landmarks_list = fr.face_landmarks(rgb_scaled, [face_locations[0]])
            if face_landmarks_list:
                landmarks = face_landmarks_list[0]
                left_eye = landmarks.get('left_eye')
                right_eye = landmarks.get('right_eye')
                
                if left_eye and right_eye:
                    # --- NEW: VISUALIZE THE AI LANDMARKS ---
                    # Draw yellow dots on the eye landmarks so reviewers can see the AI tracking
                    for point in left_eye:
                        cv2.circle(scaled, point, 2, (0, 255, 255), -1)
                    for point in right_eye:
                        cv2.circle(scaled, point, 2, (0, 255, 255), -1)

                    leftEAR = calculate_ear(left_eye)
                    rightEAR = calculate_ear(right_eye)
                    ear = (leftEAR + rightEAR) / 2.0

                    # --- NEW: DISPLAY LIVE MATH ON SCREEN ---
                    cv2.putText(scaled, f"Live EAR (Threshold 0.20): {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if ear < EYE_AR_THRESH:
                        blink_counter += 1
                        # Show visual indicator that a blink is registering
                        cv2.putText(scaled, "BLINKING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        if blink_counter >= EYE_AR_CONSEC_FRAMES:
                            liveness_passed = True
                            print("Blink detected! Liveness confirmed.")
                        blink_counter = 0

            # 2. EXTRACT ENCODING FOR IDENTITY CHECK
            try:
                face_encoding = fr.face_encodings(rgb_scaled, [(top, right, bottom, left)])[0]
                person = identifier.getIDFromEncoding(face_encoding)
                
                person_name = person
                if person and person in identifier.friendly_names:
                    person_name = identifier.friendly_names[person]

                # 3. DECISION MATRIX
                if person is None:
                    # New user logic
                    face_img = frame[(top*2):(bottom*2), (left*2):(right*2)] 
                    identifier.addNew(face_img, face_encoding)
                    accessDenied(name="New User", reason="Not Enrolled")
                    cv2.putText(scaled, "NEW USER CAPTURED", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    liveness_passed = False 
                
                elif identifier.hasAccess(person):
                    if liveness_passed:
                        accessGranted(name=person_name)
                        cv2.putText(scaled, "LIVENESS PASSED: UNLOCKED", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Briefly hold the unlock message on screen
                        ret_enc, v = cv2.imencode('.jpg', scaled)
                        identifier.setView(v.tobytes())
                        await asyncio.sleep(2) 
                        liveness_passed = False 
                    else:
                        cv2.putText(scaled, "AUTHORIZED: PLEASE BLINK", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    accessDenied(name=person_name, reason="Policy Denied")
                    cv2.putText(scaled, "ACCESS DENIED", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    liveness_passed = False 

            except IndexError:
                pass
            except Exception as e:
                print(f"Frame Error: {e}")
        else:
            liveness_passed = False
            blink_counter = 0

        # Encode and stream the view to the web dashboard EVERY FRAME
        ret_enc, v = cv2.imencode('.jpg', scaled)
        if ret_enc:
            identifier.setView(v.tobytes()) 

    vstream.release()