# Medi-Voice
Medi-Voice
# 🔊 Voice-based BioPredictor

Voice-based BioPredictor is a deep learning system that analyzes human speech and extracts biometric and health-related insights. The model uses **audio embeddings** and acoustic features to predict the following attributes from voice:

- ❤️ Heart Rate
- 🧓 Age Group (Young / Mature / Old)
- 🚻 Gender
- 🩺 Health Conditions (Multi-label prediction)
- 🔤 First Letter of Name (from speech identity cues)

---

## 🧠 Key Features

### 🎯 Multitask Prediction from Voice
- Takes a `.wav` audio file as input.
- Extracts high-level speech embeddings (e.g., ECAPA-TDNN).
- Outputs:
  - Estimated **heart rate** (regression)
  - **Age group** (classification)
  - **Gender** (binary classification)
  - Multiple **health conditions** (multi-label classification)
  - Predicted **first letter** of speaker’s name

### 🧪 Feature Extraction
- **Low-Level Acoustic Features:**
  - `meanfreq`, `sd`, `median`, `skew`, `sp.ent`, `sfm`, `mode`, `centroid`, `meanfun`, `mindom`, `maxdom`, `modindx`, etc.
- **Deep Audio Embeddings:**
  - ECAPA-TDNN embeddings for speaker and prosodic characteristics.

### 🗃️ Dataset Structure
- Custom dataset containing:
  - Audio files (.wav)
  - Metadata with labels:
    - `heart_rate` (numeric)
    - `age` (young/mature/old)
    - `gender` (male/female)
    - `health_condition_*` (binary labels for conditions)
    - `first_letter` (A-Z)

---

## 🛠️ Tech Stack

| Component      | Technology           |
|----------------|----------------------|
| Language       | Python               |
| Audio Features | `librosa`, `pyAudioAnalysis`, `torchaudio` |
| Models         | ECAPA-TDNN (Speaker Embedding), MLPs / CNNs |
| Frameworks     | PyTorch, Scikit-learn |
| Visualization  | Matplotlib, Seaborn  |

---

## 📦 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/voice-bio-predictor.git
   cd voice-bio-predictor
