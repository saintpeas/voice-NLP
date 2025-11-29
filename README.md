# Voice Recognition Suite

A modern GUI application for comparing multiple state-of-the-art voice recognition libraries side-by-side. Built with Python and CustomTkinter, this tool provides an intuitive interface for testing and evaluating different speech-to-text models.

## Features

- **Multi-Library Comparison**: Test and compare four different voice recognition engines simultaneously
  - Google Speech Recognition (Cloud-based)
  - CMU Sphinx (Offline)
  - Wav2Vec2 by Facebook/Meta
  - Whisper by OpenAI

- **Dual Input Methods**:
  - Real-time microphone recording (3-20 seconds)
  - Pre-recorded samples from LibriSpeech dataset

- **Modern UI**: Clean, dark-themed interface built with CustomTkinter
- **Asynchronous Processing**: Multi-threaded architecture for responsive UI
- **Ground Truth Comparison**: Evaluate accuracy against known transcriptions
- **Auto-save**: Recordings automatically saved with timestamps

## Prerequisites

- Python 3.8 or higher
- Microphone access (for recording features)
- Internet connection (for Google Speech Recognition and model downloads)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voice-recognition-suite.git
cd voice-recognition-suite
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**macOS**:
```bash
brew install portaudio
```

**Windows**:
No additional installation required.

## Dependencies

```
customtkinter>=5.2.0
SpeechRecognition>=3.10.0
transformers>=4.30.0
torch>=2.0.0
datasets>=2.14.0
sounddevice>=0.4.6
soundfile>=0.12.1
pocketsphinx>=5.0.0
numpy>=1.24.0

```

## Usage

### Running the application

```bash
python voice_recognition_gui.py
```

### Basic workflow

1. **Choose Input Method**:
   - Click "Start Recording" to capture audio from your microphone
   - Or click "Load Sample Audio" to use pre-recorded LibriSpeech samples

2. **Select Recognition Libraries**:
   - Check/uncheck the libraries you want to test
   - You can select any combination of the four available engines

3. **Process Audio**:
   - Click "Process Audio" to run recognition
   - Results will appear in the right panel with comparison metrics

4. **Review Results**:
   - Compare transcriptions from different libraries
   - View ground truth text (for dataset samples)
   - Analyze accuracy and performance differences

### Adjusting recording duration

Use the slider in the recording section to set duration between 3-20 seconds.

## Project Structure

```
voice-NLP/
├── voice_recogpy    # Main application file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── recordings/                  # Auto-saved recordings (created at runtime)
```

## Technical Details

### Architecture

- **GUI Framework**: CustomTkinter for modern, themed interface
- **Audio Processing**: Sounddevice for recording, Soundfile for I/O
- **Speech Recognition**: Multiple backends via transformers and SpeechRecognition libraries
- **Threading**: Background threads for model loading and audio processing

### Models

- **Wav2Vec2**: `facebook/wav2vec2-base-960h` - Pre-trained on 960 hours of LibriSpeech
- **Whisper**: `openai/whisper-tiny` - Lightweight version for faster inference
- **CMU Sphinx**: Offline recognition using PocketSphinx
- **Google**: Cloud-based recognition via SpeechRecognition API

### Dataset

LibriSpeech ASR corpus samples are loaded via HuggingFace datasets:
- Clean speech recordings
- Ground truth transcriptions included
- Multiple samples available for testing

## Test Video

### Application Demo

[https://drive.google.com/uc?export=view&id=1J4VDla_75JfRfcv_2FgwrLbQ0T33tXzE]

**Video showcases**:
- Application startup and model loading
- Recording audio from microphone
- Loading sample audio from dataset
- Processing with multiple libraries
- Comparing recognition results
- UI responsiveness and features


## Performance Considerations

- **First Run**: Initial startup will download models (approximately 500MB-1GB)
- **Model Loading**: Takes 10-30 seconds depending on hardware
- **Processing Time**: Varies by library:
  - Google: 0.5-2 seconds (network dependent)
  - Sphinx: 0.3-1 second (fastest, offline)
  - Wav2Vec2: 1-3 seconds (CPU/GPU dependent)
  - Whisper: 1-4 seconds (model size dependent)

## Troubleshooting

### Microphone not detected
- Ensure microphone permissions are granted
- Check system audio settings
- Try selecting different audio device in system preferences

### Model loading errors
- Verify internet connection for initial downloads
- Check available disk space (need ~1GB free)
- Clear HuggingFace cache: `~/.cache/huggingface/`

### Import errors
- Ensure all dependencies are installed
- Try upgrading pip: `pip install --upgrade pip`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Audio recording issues on Linux
```bash
# Install additional dependencies
sudo apt-get install python3-dev libasound2-dev
pip install pyaudio
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LibriSpeech**: ASR corpus for testing and validation
- **HuggingFace**: Transformers library and model hub
- **Facebook Research**: Wav2Vec2 model
- **OpenAI**: Whisper model
- **CMU**: Sphinx speech recognition
- **CustomTkinter**: Modern UI framework

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: dulamarklester@gmail.com

## Roadmap

- [ ] Add support for more recognition engines (DeepSpeech, Vosk)
- [ ] Implement batch processing for multiple files
- [ ] Add WER (Word Error Rate) calculation
- [ ] Export results to CSV/JSON
- [ ] Support for multiple languages
- [ ] Real-time streaming recognition
- [ ] Custom model fine-tuning interface
- [ ] Performance benchmarking dashboard

---

**Built with Python and passion for Natural Language Processing or NLP**