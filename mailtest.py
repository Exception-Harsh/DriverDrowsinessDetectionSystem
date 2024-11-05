import cv2
import numpy as np
import dlib
from pygame import mixer
from imutils import face_utils
import smtplib
from email.message import EmailMessage
import threading
from geopy.geocoders import Nominatim
from datetime import datetime
import serial

# Email configuration
EMAIL_ADDRESS = "harsh180403singh@gmail.com"
EMAIL_PASSWORD = "nwtx cbcs shwn kufx"
emergency_emails = ["harshs21ite@student.mes.ac.in"]  # List of emergency emails

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status tracking variables
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
alert_sent = False  # Flag to prevent multiple alerts

# Define GPS-based Real-Time Coordinates Retrieval
def get_real_time_gps_coordinates():
    try:
        gps = serial.Serial("COM4", baudrate=9600, timeout=1)  # Adjust for your GPS module
        while True:
            line = gps.readline().decode("ascii", errors="replace")
            if "$GPGGA" in line:
                data = line.split(",")
                if data[2] and data[4]:  # Ensure latitude and longitude are available
                    lat = float(data[2])
                    lon = float(data[4])
                    return lat, lon
    except Exception as e:
        print(f"Error reading GPS data: {e}")
        return None, None

def get_location():
    latitude, longitude = get_real_time_gps_coordinates()
    if latitude is None or longitude is None:
        return "Unable to retrieve GPS data", None
    
    try:
        geolocator = Nominatim(user_agent="drowsiness-alert")
        location = geolocator.reverse((latitude, longitude), language="en")
        return location.address if location else "Unknown location", (latitude, longitude)
    except Exception as e:
        print(f"Error obtaining location: {e}")
        return "Error obtaining location", None

# Function to send alert email with location
def send_email_alert():
    address, coordinates = get_location()
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email_content = f"""
    Alert: The driver appears to be drowsy.

    Location: {address}
    Coordinates: {coordinates}
    Time: {time_now}
    Immediate attention may be needed.
    """

    msg = EmailMessage()
    msg.set_content(email_content)
    msg["Subject"] = "Drowsiness Alert with Location!"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = emergency_emails

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email alert sent to emergency contacts.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Functions for drowsiness detection
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    return 2 if ratio > 0.25 else 1 if ratio > 0.21 else 0

def play_alert_sound():
    mixer.init()
    mixer.music.load("alarm2.mp3")  # Ensure correct path
    mixer.music.set_volume(1.0)    # Set volume to max
    mixer.music.play()    

# Main loop for drowsiness detection
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_frame = frame.copy()
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 15:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if not alert_sent:
                    threading.Thread(target=play_alert_sound).start()
                    threading.Thread(target=send_email_alert).start()
                    alert_sent = True

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                alert_sent = False  # Reset alert if the driver becomes active

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
        break 

cap.release()
cv2.destroyAllWindows()
