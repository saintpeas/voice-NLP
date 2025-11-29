import customtkinter as ctk
from tkinter import messagebox
import threading
import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import numpy as np
from datasets import load_dataset
import sounddevice as sd
import soundfile as sf
from datetime import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class VoiceRecognitionGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🎤 Voice Recognition Suite")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.is_recording = False
        self.current_audio = None
        self.models_loaded = False
        
        self.create_widgets()
        self.load_models_thread()
        
    def create_widgets(self):
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        
        title = ctk.CTkLabel(
            header,
            text="🎤 Voice Recognition Suite",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title.pack(side="left")
        
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.create_recording_section(left_frame)
        self.create_sample_section(left_frame)
        self.create_library_section(left_frame)
        
        right_frame = ctk.CTkFrame(main_container)
        right_frame.pack(side="right", fill="both", expand=True)
        
        self.create_results_section(right_frame)
        
        self.status_label = ctk.CTkLabel(
            self.root,
            text="Ready - Loading models...",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="bottom", fill="x", padx=20, pady=10)
        
    def create_recording_section(self, parent):
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=10, pady=10)
        
        label = ctk.CTkLabel(
            section,
            text="🎙️ Record Audio",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        label.pack(pady=(10, 5))
        
        self.record_btn = ctk.CTkButton(
            section,
            text="⏺ Start Recording",
            command=self.toggle_recording,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b"
        )
        self.record_btn.pack(pady=10, padx=20, fill="x")
        
        duration_label = ctk.CTkLabel(section, text="Recording Duration (seconds):")
        duration_label.pack(pady=(10, 0))
        
        self.duration_slider = ctk.CTkSlider(
            section,
            from_=3,
            to=20,
            number_of_steps=17
        )
        self.duration_slider.set(10)
        self.duration_slider.pack(pady=5, padx=20, fill="x")
        
        self.duration_value = ctk.CTkLabel(section, text="10 seconds")
        self.duration_value.pack()
        
        self.duration_slider.configure(command=self.update_duration_label)
        
    def create_sample_section(self, parent):
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=10, pady=10)
        
        label = ctk.CTkLabel(
            section,
            text="📚 Load Sample from Dataset",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        label.pack(pady=(10, 5))
        
        info = ctk.CTkLabel(
            section,
            text="Load pre-recorded audio from LibriSpeech",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        info.pack()
        
        self.load_sample_btn = ctk.CTkButton(
            section,
            text="📥 Load Sample Audio",
            command=self.load_sample_audio,
            height=40
        )
        self.load_sample_btn.pack(pady=10, padx=20, fill="x")
        
        self.sample_var = ctk.StringVar(value="Sample 1")
        sample_menu = ctk.CTkOptionMenu(
            section,
            values=["Sample 1", "Sample 2", "Sample 3"],
            variable=self.sample_var
        )
        sample_menu.pack(pady=(0, 10), padx=20, fill="x")
        
    def create_library_section(self, parent):
        section = ctk.CTkFrame(parent)
        section.pack(fill="both", expand=True, padx=10, pady=10)
        
        label = ctk.CTkLabel(
            section,
            text="🔧 Recognition Libraries",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        label.pack(pady=(10, 5))
        
        self.use_google = ctk.CTkCheckBox(
            section,
            text="Google Speech Recognition",
            font=ctk.CTkFont(size=13)
        )
        self.use_google.select()
        self.use_google.pack(pady=5, padx=20, anchor="w")
        
        self.use_sphinx = ctk.CTkCheckBox(
            section,
            text="CMU Sphinx (Offline)",
            font=ctk.CTkFont(size=13)
        )
        self.use_sphinx.select()
        self.use_sphinx.pack(pady=5, padx=20, anchor="w")
        
        self.use_wav2vec = ctk.CTkCheckBox(
            section,
            text="Wav2Vec2 (Facebook)",
            font=ctk.CTkFont(size=13)
        )
        self.use_wav2vec.select()
        self.use_wav2vec.pack(pady=5, padx=20, anchor="w")
        
        self.use_whisper = ctk.CTkCheckBox(
            section,
            text="Whisper (OpenAI)",
            font=ctk.CTkFont(size=13)
        )
        self.use_whisper.pack(pady=5, padx=20, anchor="w")
        
        self.process_btn = ctk.CTkButton(
            section,
            text="🚀 Process Audio",
            command=self.process_audio,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#27ae60",
            hover_color="#229954"
        )
        self.process_btn.pack(pady=20, padx=20, fill="x")
        
    def create_results_section(self, parent):
        label = ctk.CTkLabel(
            parent,
            text="📊 Recognition Results",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        label.pack(pady=10)
        
        self.results_text = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(size=13),
            wrap="word"
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        clear_btn = ctk.CTkButton(
            parent,
            text="🗑️ Clear Results",
            command=self.clear_results,
            height=35
        )
        clear_btn.pack(pady=(0, 10), padx=10, fill="x")
        
    def update_duration_label(self, value):
        self.duration_value.configure(text=f"{int(value)} seconds")
        
    def update_status(self, message):
        self.status_label.configure(text=message)
        self.root.update()
        
    def load_models_thread(self):
        def load():
            try:
                self.update_status("Loading Wav2Vec2 model...")
                self.recognizer = sr.Recognizer()
                
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                
                self.update_status("Loading Whisper model...")
                self.whisper_pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny"
                )
                
                self.models_loaded = True
                self.update_status("✅ Ready - All models loaded!")
                
            except Exception as e:
                self.update_status(f"⚠️ Error loading models: {str(e)}")
                messagebox.showerror("Error", f"Failed to load models: {str(e)}")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.is_recording = True
        self.record_btn.configure(
            text="⏹ Stop Recording",
            fg_color="#e74c3c"
        )
        self.update_status("🔴 Recording...")
        
        def record():
            try:
                duration = int(self.duration_slider.get())
                sample_rate = 16000
                
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                
                self.current_audio = recording.flatten()
                self.is_recording = False
                
                self.root.after(0, lambda: self.record_btn.configure(
                    text="⏺ Start Recording",
                    fg_color="#27ae60"
                ))
                self.root.after(0, lambda: self.update_status("✅ Recording complete!"))
                
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                sf.write(filename, recording, sample_rate)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Recording failed: {str(e)}"
                ))
                self.is_recording = False
                
        thread = threading.Thread(target=record, daemon=True)
        thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        sd.stop()
        
    def load_sample_audio(self):
        self.update_status("📥 Loading sample from LibriSpeech...")
        
        def load():
            try:
                dataset = load_dataset(
                    "hf-internal-testing/librispeech_asr_dummy",
                    "clean",
                    split="validation"
                )
                
                sample_idx = int(self.sample_var.get().split()[-1]) - 1
                sample = dataset[sample_idx]
                
                self.current_audio = sample['audio']['array']
                ground_truth = sample['text']
                
                self.root.after(0, lambda: self.update_status(
                    f"✅ Sample loaded! Ground truth: {ground_truth}"
                ))
                self.root.after(0, lambda: self.append_result(
                    f"📝 Ground Truth Text:\n{ground_truth}\n\n" + "="*50 + "\n\n"
                ))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to load sample: {str(e)}"
                ))
                
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def process_audio(self):
        if self.current_audio is None:
            messagebox.showwarning("Warning", "Please record or load audio first!")
            return
            
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait.")
            return
            
        self.clear_results()
        self.update_status("🔄 Processing audio...")
        self.process_btn.configure(state="disabled")
        
        def process():
            try:
                results = []
                
                if self.use_google.get():
                    result = self.recognize_google()
                    if result:
                        results.append(("🌐 Google Speech Recognition", result))
                
                if self.use_sphinx.get():
                    result = self.recognize_sphinx()
                    if result:
                        results.append(("🔒 CMU Sphinx (Offline)", result))
                
                if self.use_wav2vec.get():
                    result = self.recognize_wav2vec()
                    if result:
                        results.append(("🤖 Wav2Vec2 (Facebook)", result))
                
                if self.use_whisper.get():
                    result = self.recognize_whisper()
                    if result:
                        results.append(("🎯 Whisper (OpenAI)", result))
                
                for name, text in results:
                    self.root.after(0, lambda n=name, t=text: self.append_result(
                        f"{n}:\n{t}\n\n" + "="*50 + "\n\n"
                    ))
                
                self.root.after(0, lambda: self.update_status("✅ Processing complete!"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Processing failed: {str(e)}"
                ))
            finally:
                self.root.after(0, lambda: self.process_btn.configure(state="normal"))
                
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
        
    def recognize_google(self):
        try:
            audio_int16 = (self.current_audio * 32767).astype(np.int16)
            audio_data = sr.AudioData(audio_int16.tobytes(), 16000, 2)
            return self.recognizer.recognize_google(audio_data)
        except Exception as e:
            return f"Error: {str(e)}"
            
    def recognize_sphinx(self):
        try:
            audio_int16 = (self.current_audio * 32767).astype(np.int16)
            audio_data = sr.AudioData(audio_int16.tobytes(), 16000, 2)
            return self.recognizer.recognize_sphinx(audio_data)
        except Exception as e:
            return f"Error: {str(e)}"
            
    def recognize_wav2vec(self):
        try:
            inputs = self.wav2vec_processor(
                self.current_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                logits = self.wav2vec_model(inputs.input_values).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.wav2vec_processor.batch_decode(predicted_ids)[0]
        except Exception as e:
            return f"Error: {str(e)}"
            
    def recognize_whisper(self):
        try:
            result = self.whisper_pipe(self.current_audio, sampling_rate=16000)
            return result['text']
        except Exception as e:
            return f"Error: {str(e)}"
            
    def append_result(self, text):
        self.results_text.insert("end", text)
        self.results_text.see("end")
        
    def clear_results(self):
        self.results_text.delete("1.0", "end")
        
    def run(self):
        self.root.mainloop()


def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║          Voice Recognition GUI Application                     ║
    ║                                                                ║
    ║  Libraries Used:                                               ║
    ║  • CustomTkinter - Modern GUI                                  ║
    ║  • SpeechRecognition - Google & Sphinx                        ║
    ║  • Transformers - Wav2Vec2 & Whisper                          ║
    ║  • Datasets - LibriSpeech samples                             ║
    ║  • Sounddevice - Audio recording                              ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    app = VoiceRecognitionGUI()
    app.run()


if __name__ == "__main__":
    main()