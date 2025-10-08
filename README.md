# 🎵 Emotion2Music

Transform your emotions into music! This application uses deep learning to analyze your emotional keywords and find the perfect matching music.

## ✨ Features

- 🎭 **Emotion Analysis**: Convert text keywords to emotional dimensions (Valence, Arousal, Dominance)
- 💫 **Interactive Bubble Interface**: Add keywords as animated bubbles with smooth transitions
- 🎵 **Music Matching**: Find songs that match your emotional state
- 🥁 **BPM Prediction**: Predict the tempo that fits your mood
- 🎧 **Instant Playback**: Stream music directly in the browser
- 🎨 **Beautiful UI**: Modern, gradient-based interface with drag-and-drop visual metaphor
- 🎯 **Preset Emotions**: Quick-select from 8 common emotion combinations
- ✨ **One Keyword at a Time**: Add emotion keywords individually for precise control

## 🚀 Quick Start

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

## 🌐 Deploy to HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - Copy the `model/` folder with trained models
   - Copy the `data/` folder with dataset

4. Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/emotion2music`

## 📁 Required Files

```
Emotion2Music/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model/
│   └── va_2d_model.pth            # Trained model
└── data/
    └── top3_themes_with_vad_mood_900.tsv  # Dataset
```

## 🎯 How It Works

1. **Input**: User adds emotional keywords one at a time (e.g., "happy", "energetic", "dancing")
   - Keywords appear as animated bubbles in the Emotion Area
   - Users can add keywords individually or select preset combinations
2. **VAD Conversion**: Keywords are converted to Valence-Arousal-Dominance values using NRC-VAD lexicon
3. **Prediction**: Neural network predicts mood category and BPM
4. **Retrieval**: System finds the best matching song from database
5. **Playback**: Audio streams from Jamendo API

## 🎨 Model Architecture

- **Input**: 2D (Valence, Arousal) or 3D (Valence, Arousal, Dominance)
- **Architecture**: Multi-layer perceptron with dual heads
  - Classification head: Predicts mood category
  - Regression head: Predicts BPM
- **Training**: Multi-task learning with combined loss

## 🎵 Supported Moods

The model can predict various moods including:
- Happy, Sad, Calm, Energetic
- Dark, Romantic, Aggressive
- And more...

## 📊 Dataset

- **Source**: MTG-Jamendo Dataset
- **Size**: 900+ annotated tracks
- **Features**: Mood tags, BPM, VAD values

## 🤝 Credits

- **Dataset**: MTG-Jamendo Dataset
- **Lexicon**: NRC-VAD-Lexicon
- **Music API**: Jamendo API
- **Framework**: Streamlit, PyTorch

## 📝 License

This project is for educational purposes. Please check the licenses of individual components:
- MTG-Jamendo Dataset: [License](https://mtg.github.io/mtg-jamendo-dataset/)
- Jamendo Music: Creative Commons licensed tracks

Made with ❤️ using Streamlit, PyTorch & Jamendo API

