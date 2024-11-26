# Transcribe Audio

**Transcribe Audio** is a Python application designed to simplify the transcription of audio files. It utilizes advanced machine learning models like Whisper and Pyannote for transcription and speaker diarization, providing accurate and formatted outputs.

---

## Features
- **Audio Splitting**: Automatically segments large audio files into smaller parts for better processing.
- **Speaker Diarization**: Identifies and assigns roles (e.g., interviewer, interviewee) to different speakers in the audio.
- **Transcription Quality Options**: Choose between advanced, medium, and fast transcription modes depending on the audio length and desired accuracy.
- **Graphical User Interface**: An easy-to-use interface with progress tracking.
- **Compression**: Automatically compresses transcription files into a `.zip` for convenient sharing.

---

## Why This Code Structure?
1. **Efficiency**: The application splits audio into manageable segments, ensuring smooth processing even for long recordings.
2. **Advanced Models**: 
   - **Whisper**: Provides robust transcription capabilities.
   - **Pyannote**: Adds speaker diarization to differentiate between speakers.
3. **Modularity**: Functions like `split_audio`, `diarize_audio`, and `transcribe_audio` are designed to be reusable and independent.
4. **User-Friendly GUI**: Built with `ttkbootstrap`, the interface makes it accessible even for non-technical users.
5. **Cross-Platform Compatibility**: Supports both Windows and Linux environments with minimal dependencies.

---

## How It Works
1. **Audio Splitting**:
   - The `split_audio` function uses `ffmpeg` to divide audio files into smaller chunks.
2. **Speaker Diarization**:
   - The `diarize_audio` function employs Pyannote models to determine speaker turns and roles.
3. **Transcription**:
   - Whisper processes each segment to generate text, which is then formatted with speaker annotations.
4. **Compression**:
   - Transcriptions are optionally compressed into a `.zip` file for easy sharing.

---
