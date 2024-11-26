import os
import whisper
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pyannote.audio import Pipeline
import torch
import time
from threading import Thread
import ttkbootstrap as tb
from zipfile import ZipFile

def split_audio(audio_path, output_dir, segment_duration):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c", "copy",
        os.path.join(output_dir, "part%03d.mp3")
    ]
    subprocess.run(command, check=True)

def diarize_audio(audio_path):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="SEU_TOKEN_HF")
        diarization = pipeline(audio_path)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        return speaker_segments
    except Exception as e:
        print(f"Erro na diarização: {e}")
        return None

def assign_roles(speaker_segments):
    roles = {}
    for i, segment in enumerate(speaker_segments):
        if i == 0:
            roles[segment["speaker"]] = "Entrevistador"
        elif i == 1:
            roles[segment["speaker"]] = "Entrevistada"
        else:
            roles[segment["speaker"]] = f"Participante {i - 1}"
    return roles

def format_transcription(result, speaker_segments=None):
    segments = result.get("segments", [])
    formatted_text = []
    roles = assign_roles(speaker_segments) if speaker_segments else {}

    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        speaker = "Desconhecido"
        if speaker_segments:
            for speaker_segment in speaker_segments:
                if speaker_segment["start"] <= start_time < speaker_segment["end"]:
                    speaker = roles.get(speaker_segment["speaker"], "Desconhecido")
                    break

        formatted_text.append(f"[{start_time:.2f}s - {end_time:.2f}s] ({speaker}): {text}")
    return "\n\n".join(formatted_text)

def transcribe_audio(audio_path, output_base, model, quality):
    try:
        segment_duration = {"advanced": 300, "medium": 600, "fast": 900}[quality]
        segments_dir = "audio_segments"
        split_audio(audio_path, segments_dir, segment_duration)

        for idx, segment_file in enumerate(sorted(os.listdir(segments_dir))):
            segment_path = os.path.join(segments_dir, segment_file)
            output_path = f"{output_base}_part{idx + 1:03d}.txt"

            result = model.transcribe(segment_path, task="transcribe")
            speaker_segments = diarize_audio(segment_path)
            formatted_text = format_transcription(result, speaker_segments)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(formatted_text)
    except Exception as e:
        print(f"Erro ao processar áudio: {e}")

def compress_output(output_dir):
    zip_path = os.path.join(output_dir, "transcricoes.zip")
    with ZipFile(zip_path, "w") as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".txt"):
                    zipf.write(os.path.join(root, file), arcname=file)
    messagebox.showinfo("Compactação Concluída", f"Arquivos compactados em: {zip_path}")

def open_output_folder(output_dir):
    os.startfile(output_dir)

# Interface Gráfica
def start_transcription(quality, root):
    def run():
        loading_label.config(text="Carregando...")
        progress_bar.start()

        audio_file = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav")])
        if not audio_file:
            progress_bar.stop()
            loading_label.config(text="Nenhum arquivo selecionado.")
            return

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, "transcricao")

        model = whisper.load_model("large" if quality == "advanced" else "small", device="cuda" if torch.cuda.is_available() else "cpu")

        transcribe_audio(audio_file, output_base, model, quality)

        progress_bar.stop()
        loading_label.config(text="Processamento concluído.")
        open_output_folder(output_dir)
    
    Thread(target=run).start()

def create_gui():
    root = tb.Window(themename="superhero")
    root.title("Transcrição de Áudio")
    root.geometry("500x500")
    root.iconbitmap("Icon/icon.ico")

    frame = ttk.Frame(root, padding=20)
    frame.pack(expand=True, fill="both")

    title = tb.Label(frame, text="Escolha a qualidade da transcrição:", font=("Helvetica", 16), anchor="center")
    title.pack(pady=20)

    advanced_button = tb.Button(frame, text="Avançada (áudios curtos)", bootstyle="primary-outline", command=lambda: start_transcription("advanced", root))
    advanced_button.pack(pady=10, ipadx=10, ipady=5)

    medium_button = tb.Button(frame, text="Média (áudios médios)", bootstyle="success-outline", command=lambda: start_transcription("medium", root))
    medium_button.pack(pady=10, ipadx=10, ipady=5)

    fast_button = tb.Button(frame, text="Rápida (áudios longos)", bootstyle="warning-outline", command=lambda: start_transcription("fast", root))
    fast_button.pack(pady=10, ipadx=10, ipady=5)

    compress_button = tb.Button(frame, text="Compactar Transcrições", bootstyle="info", command=lambda: compress_output("output"))
    compress_button.pack(pady=10, ipadx=10, ipady=5)

    global loading_label, progress_bar
    loading_label = tb.Label(frame, text="", font=("Helvetica", 12))
    loading_label.pack(pady=10)

    progress_bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="indeterminate")
    progress_bar.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
