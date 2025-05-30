# 🧠 Face Recognition AI Assistant
<p align="justify">
This project implements a <strong>secure face-authenticated AI assistant</strong> using <strong>OpenCV</strong> and <strong>Google Gemini API</strong>. It combines computer vision and speech technologies to create a hands-free assistant that only works after verifying your identity. To ensure portability and ease of use, the assistant can also be packaged as a <strong>standalone .exe</strong> file that runs on any Windows machine without needing a Python environment. <strong>You can store this .exe on a USB drive and carry it with you</strong>, allowing you to access your <strong>personalized assistant</strong> from any computer - without needing to log in or set up anything on someone else’s PC.
</p>

# 📊 Problem Statement
<p align="justify">
Most <strong>digital assistants</strong> are either <strong>unsecured</strong> or<strong> tied to specific devices and user accounts</strong>, making them inconvenient for <strong>personalized use across multiple systems</strong>. Logging into someone else's PC or setting up environments can be <strong>time-consuming</strong> and <strong>risky</strong>. This project solves that by combining <strong>face recognition-based authentication</strong> with a <strong>portable AI assistant</strong> that runs directly from a USB drive. It ensures that <strong>only you</strong> can access your assistant and preferences on <strong>any Windows machine</strong> without leaving traces or requiring installation.
</p>

# 🔍 Key Features
- **👤 Face-Based Access**: Ensures secure interaction by allowing only authenticated users via webcam.
- **🔊 Voice Assistant**: Talks and listens using `pyttsx3` and `SpeechRecognition`.  
- **🤖 Gemini AI Integration**: Uses Google's Gemini model to answer natural language questions.  
- **📦 Executable Support**: Includes a PyInstaller-compatible script (`PORTABLE.py`) to generate a `.exe`.  
- **🧪 Workflow Interface**: Menu-driven interface to collect samples, train models, and launch the assistant.  

# 💡 Technologies Used
- **Python**: Core language.
- **Pillow**, **NumPy**: Image handling and math. 
- **OpenCV**: Face detection and recognition.
- **SpeechRecognition**: Captures and transcribes speech.    
- **pyttsx3**: Converting AI responses to speech.  
- **Google Generative AI (Gemini API)**: Responds intelligently to user queries.  
- **PyAutoGUI**: UI automation utility.
- **PyInstaller**: Packaging the project into a standalone .exe file for portable use.

# 📦 Repository Contents
| Folder / File       | Description |
|---------------------|-------------|
| `1-DATA/`           | Haarcascade XML file for detecting faces. |
| `2-SAMPLES/`        | Captured face image samples. |
| `3-TRAINER/`        | Trained LBPH model (`Trainer.yml`). |
| `4-INSTRUCTIONS.txt`| Step-by-step guide to build the .exe file. |
| `REQUIREMENTS.txt`  | List of Python dependencies. |
| `AI ASSISTANT.py`   | Full script to capture, train, recognize, and run assistant. |
| `PORTABLE.py`       | Lightweight PyInstaller-ready launcher. |

# 🚀 Running the App Locally
```bash
# 1. Clone the repository
git clone https://github.com/Noisy-Debug/Face-Recognition-AI-Assistant.git

# 2. Install dependencies
pip install -r REQUIREMENTS.txt

# 3. Launch the Assistant
python "AI ASSISTANT.py"
```

# ⚙️ Convert to Executable
```bash
pyinstaller --onefile --add-data "1-DATA;1-DATA" --add-data "3-TRAINER;3-TRAINER" PORTABLE.py
```
For Full Setup and Cleanup Steps, See 4-INSTRUCTIONS.txt

# 📊 Results and Insights
- **LBPH** Face Recognizer ensures fast and accurate authentication.  
- **Gemini Model** provides intelligent responses with natural language understanding.  
- **Real-time voice-based interaction** without **manual typing**. 
- **Portable access** across any system with **USB support**.

# 🔮 Future Enhancements
- Support for **multiple users and dynamic face labels**.  
- Add **dashboard** for managing training data and usage logs.  
- Enable **continuous learning** from new face data.  
- Implement **error logging** and offline fallback mechanisms.  
- Integrate **Explainable AI (XAI)** for Gemini responses.