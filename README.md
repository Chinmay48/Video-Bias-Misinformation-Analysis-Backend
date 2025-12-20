# 🎥 Video Bias & Misinformation Analysis Backend

An end-to-end AI backend system that analyzes educational videos to detect **bias, emotional manipulation, and misinformation** using a fully automated pipeline built with **FastAPI** and **open-source AI models**.

The system accepts a **YouTube link or uploaded video**, extracts all possible textual content (speech + on-screen text), and evaluates the **credibility and bias** of the video.

---

## 🚀 Key Features

### ✅ Multi-Input Support
- 📺 YouTube video links  
- 🔗 Direct video URLs  
- 📁 Uploaded video files  

### ✅ Smart Transcript Handling
- Uses **YouTube Transcript API** when captions are available (fast & free)
- Automatically falls back to **Whisper (Speech-to-Text)** when captions are unavailable

### ✅ OCR-Based On-Screen Text Extraction
- Extracts frames using **OpenCV**
- Reads text from slides, code, diagrams using **EasyOCR / Tesseract**

### ✅ Advanced NLP Analysis
- Text cleaning & preprocessing using **spaCy**
- Sentence-level analysis

### ✅ Bias Detection
- Emotional tone analysis
- Subjectivity & opinion detection
- Manipulative language detection
- Bias scoring using **HuggingFace models**

### ✅ Misinformation Detection
- Extracts factual claims
- Cross-checks claims using public knowledge sources (Wikipedia-style approach)
- Labels content as:
  - ✔ Supported
  - ❌ Contradicted
  - ⚠ Uncertain

### ✅ Clean API Output
- Transcript
- OCR text
- Cleaned text
- Bias & misinformation report (JSON)

---

## 🧠 Tech Stack

### Backend
- **FastAPI** – API framework  
- **Uvicorn** – ASGI server  

### AI / ML
- **Whisper (OpenAI)** – Speech-to-text  
- **spaCy** – NLP preprocessing  
- **HuggingFace Transformers** – Bias, sentiment, subjectivity models  
- **EasyOCR / Tesseract** – OCR  
- **OpenCV** – Frame extraction  

### Video & Audio
- **yt-dlp** – Video downloading  
- **FFmpeg** – Audio extraction  

---

## 📂 Project Folder Structure
MisinformationVideoPart/<br>
│<br>
├── app/<br>
│   ├── main.py                     # FastAPI app entrypoint<br>
│   <br>
│   ├── api/<br>
│   │   └── routes/<br>
│   │       ├── analyze_video.py    # Main analysis endpoint<br>
│   │       └── health.py           # Health check endpoint<br>
│<br>
│   ├── models/<br>
│   │   ├── request_models.py       # API request schemas<br>
│   │   └── response_models.py      # API response schemas<br>
│   <br>
│   ├── pipeline/<br>
│   │   └── run_pipeline.py         # Orchestrates entire workflow<br>
│<br>
│   ├── services/<br>
│   │   ├── input_handler/<br>
│   │   │   ├── detect_input_type.py  # Detects YouTube / URL / File<br>
│   │   │   ├── download_video.py     # Downloads video using yt-dlp<br>
│   │   │   └── extract_audio.py      # Extracts audio using FFmpeg<br>
│   │   │<br>
│   │   ├── transcript/<br>
│   │   │   ├── youtube_transcript.py # Fetches YouTube captions<br>
│   │   │   └── whisper_transcript.py # Whisper speech-to-text<br>
│   │   │<br>
│   │   ├── ocr/<br>
│   │   │   ├── frame_extractor.py    # Extracts frames via OpenCV<br>
│   │   │   └── ocr_reader.py         # OCR using EasyOCR/Tesseract<br>
│   │   │<br>
│   │   ├── nlp/<br>
│   │   │   ├── text_preprocessing.py # Cleaning & sentence splitting<br>
│   │   │   ├── merge_text.py         # Merge transcript + OCR text<br>
│   │   │   ├── bias_detection.py     # Bias & opinion detection<br>
│   │   │   └── misinformation_detection.py # Fact checking<br>
│   │   │<br>
│   │   └── utils/<br>
│   │       ├── file_utils.py         # Temp file cleanup<br>
│   │       ├── logger.py             # Logging utilities<br>
│   │       └── constants.py          # Constants & configs<br>
│<br>
├── temp_files/                      # Temporary video/audio/frame storage<br>
│<br>
├── tests/                           # Unit tests<br>
│<br>
├── requirements.txt                 # Project dependencies<br>
├── README.md                        # Project documentation<br>
└── .gitignore                       # Git ignore rules<br>


---

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/video-bias-backend.git
cd video-bias-backend
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```
### 5️⃣ Install FFmpeg (Required)
```bash
Download FFmpeg from: https://ffmpeg.org/download.html
```
###🧩 Step 1: Download FFmpeg
```bash
Open: 👉 https://ffmpeg.org/download.html

Click Windows

Click Windows builds by gyan.dev

Download:

ffmpeg-release-essentials.zip

```


### 📂 Step 2: Extract FFmpeg
```bash
Right-click the downloaded ZIP file

Click Extract All

Move the extracted folder to:

C:\ffmpeg


Your folder should look like:

C:\ffmpeg
 └── bin
     ├── ffmpeg.exe
     ├── ffprobe.exe
     └── ffplay.exe
```
### ⚙️ Step 3: Add FFmpeg to PATH
```bash
Press Windows + S

Search: Environment Variables

Click:

Edit the system environment variables


Click Environment Variables
```
### ➕ Step 4: Edit PATH Variable
```bash
For User PATH (recommended)
Under User variables, select:
Path
Click Edit
Click New
Paste:
C:\ffmpeg\bin
Click OK → OK → OK
```
### 🔁 Step 5: Restart Terminal
```bash
⚠️ Important:
Close PowerShell / CMD / VS Code completely and reopen it.

### ✅ Step 6: Verify Installation

Open a new terminal and run:

ffmpeg -version


If installed correctly, you’ll see output like:

ffmpeg version 6.x ...
built with gcc ...
```

### ▶️ Running the Application
```bash
uvicorn app.main:app --reload
```

Open in browser:

📘 API Docs: http://127.0.0.1:8000/docs

❤️ Health Check: http://127.0.0.1:8000/health
