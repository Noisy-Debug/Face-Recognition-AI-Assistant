import cv2
import re
import numpy as np
from PIL import Image
import os
import pyttsx3
import speech_recognition as sr
import datetime
import pyautogui as p
import google.generativeai as genai

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
# Step 1: Collect Face Samples
# ============================================== #
def collect_samples():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 1280)
    cam.set(4, 720)

    detector = cv2.CascadeClassifier('1-DATA/Haarcascade-Frontalface-Default.xml')
    if detector.empty():
        print("Error: Haarcascade file not loaded correctly.")
        return

    face_id = input("Enter a Numeric user ID: ")
    print("Taking samples, look at the camera...")
    
    os.makedirs("2-SAMPLES", exist_ok=True)
    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"2-SAMPLES/Face.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.imshow('image', img)

        if cv2.waitKey(100) & 0xff == 27 or count >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

# ============================================== #
# Step 2: Train the Face Recognizer
# ============================================== #
def train_model():
    path = '2-SAMPLES'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('1-DATA/Haarcascade-Frontalface-Default.xml')

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for imagePath in image_paths:
            gray_img = Image.open(imagePath).convert('L')
            img_arr = np.array(gray_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_arr)
            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)

        return face_samples, ids

    faces, ids = get_images_and_labels(path)
    if not faces:
        print("No faces detected during training.")
        return

    os.makedirs('3-TRAINER', exist_ok=True)
    recognizer.train(faces, np.array(ids))
    recognizer.write(f'3-TRAINER/Trainer.yml')

# ============================================== #
# Step 3: Recognize and Launch Nova
# ============================================== #
def recognize_and_execute():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f'3-TRAINER/Trainer.yml')
    faceCascade = cv2.CascadeClassifier('1-DATA/Haarcascade-Frontalface-Default.xml')

    font = cv2.FONT_HERSHEY_SIMPLEX
    names = ['', 'Tejas']

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    CONFIDENCE_THRESHOLD = 65  # Lower is better for LBPH
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
# Main Menu Interface
# ============================================== #
if __name__ == "__main__":
    while True:
        print("\n===== Nova Face Recognition Assistant =====")
        print("1. Collect Face Samples")
        print("2. Train Model")
        print("3. Recognize and Launch Nova")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            print("\n[INFO] Capturing image...")
            collect_samples()
            print("✅ Done capturing.")
            next_step = input("👉 Do you want to train the model now? (y/n): ").lower()
            if next_step == 'y':
                print("\n[INFO] Training model...")
                train_model()
                print("✅ Model training complete.")
                next_launch = input("👉 Do you want to launch Nova now? (y/n): ").lower()
                if next_launch == 'y':
                    print("\n[INFO] Launching Nova...")
                    recognize_and_execute()
                elif next_launch == 'n':
                    print("👍 Skipping Nova launch. Returning to main menu.")
                else:
                    print("❌ Invalid input. Returning to main menu.")
            elif next_step == 'n':
                print("👍 Skipping training. Returning to main menu.")
            else:
                print("❌ Invalid input. Returning to main menu.")

        elif choice == '2':
            if not any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir('2-SAMPLES')):
                print("❌ No image samples found. Please run step 1 first.")
            else:
                print("\n[INFO] Training model...")
                train_model()
                print("✅ Model training complete.")
                next_launch = input("👉 Do you want to launch Nova now? (y/n): ").lower()
                if next_launch == 'y':
                    print("\n[INFO] Launching Nova...")
                    recognize_and_execute()
                elif next_launch == 'n':
                    print("👍 Skipping Nova launch. Returning to main menu.")
                else:
                    print("❌ Invalid input. Returning to main menu.")

        elif choice == '3':
            if not os.path.exists('3-TRAINER/Trainer.yml'):
                print("❌ Trainer file not found. Please run step 2 first.")
            else:
                print("\n[INFO] Launching Nova...")
                recognize_and_execute()

        elif choice == '4':
            engine.stop()
            del engine
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid input. Please enter 1, 2, 3, or 4.")
