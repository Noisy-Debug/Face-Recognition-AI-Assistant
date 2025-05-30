import cv2
import re
import numpy as np
import os
from PIL import Image
import pyttsx3
import speech_recognition as sr
import datetime
import pyautogui as p
import google.generativeai as genai
import sys

def resource_path(relative_path):
    """ Get absolute path to resource (for PyInstaller) """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ============================================== #
# Gemini + Voice Engine Initialization
# ============================================== #
genai.configure(api_key="Key Required") # Replace with your personal key

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(text):
    cleaned = re.sub(r'[*_~>#]', '', text)
    print(f"Nova: {cleaned}")
    try:
        engine.say(cleaned)
        engine.runAndWait()
    except Exception as e:
        print(f"[Speech Error Ignored] {e}")

def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Sorry, I couldn't get a response from Gemini."

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour < 12:
        speak("Good Morning")
    elif hour < 18:
        speak("Good Afternoon")
    else:
        speak("Good Evening")
    speak("Hi I am Nova. How can I help you?")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User Said: {query}\n")
    except:
        speak("Say that again please...")
        return "none"
    return query.lower()

def TaskExecution():
    p.press('esc')
    speak("Verification Successful. Welcome back Tejas.")
    wishMe()
    try:
        while True:
            query = takeCommand()
            if 'exit' in query:
                speak("Goodbye.")
                break
            elif query != "none":
                speak("Let me think...")
                ai_response = ask_gemini(query)
                speak(ai_response)
    except KeyboardInterrupt:
        speak("Keyboard interrupt received. Exiting now.")

# ============================================== #
# Face Recognition and Launch
# ============================================== #
def recognize_and_execute():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(resource_path('3-TRAINER/Trainer.yml'))
    faceCascade = cv2.CascadeClassifier(resource_path('1-DATA/Haarcascade-Frontalface-Default.xml'))

    names = ['', 'Tejas']
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    CONFIDENCE_THRESHOLD = 65
    authenticated = False
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to access camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, accuracy = recognizer.predict(gray[y:y + h, x:x + w])

            if accuracy < CONFIDENCE_THRESHOLD:
                id_name = names[id] if id < len(names) else "User"
                authenticated = True
                speak(f"Hello {id_name}")
                TaskExecution()
                break
            else:
                speak("User authentication failed.")
                break

        cv2.imshow('camera', img)
        if cv2.waitKey(10) & 0xff == 27 or authenticated:
            break

    if not authenticated:
        speak("User authentication failed. Please try again.")

    cam.release()
    cv2.destroyAllWindows()

# ============================================== #
# Auto Launch Nova
# ============================================== #
if __name__ == "__main__":
    print("\n[INFO] Launching Nova...")
    if not os.path.exists(resource_path('3-TRAINER/Trainer.yml')):
        print("❌ Model Not Found. Please Follow Instruction File Setup Correctly.")
    else:
        recognize_and_execute()