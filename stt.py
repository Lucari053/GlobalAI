import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time
import threading
from constants import *

class SpeechTranscriber:
    def __init__(self, new_prompt, stop_generation):
        # Initialisation du model et de variable editable
        self.model = WhisperModel("base", device="cuda")
        self.tick_time = STT_TickTime
        self.silence_threshold = STT_silence_threshold
        self.silence_timeout = STT_silence_timeout
        self.new_prompt = new_prompt
        self.stop_generation = stop_generation

        # Initialisation de variable random
        self.audio_data = np.empty((0, 1), dtype=np.float32)
        self.block_data = np.empty((0, 1), dtype=np.float32)
        self.block_data_time = time.time()
        self.is_recording = False


    def audio_callback(self, indata, frames, _, status):
        """
        Appeller chaque tick pour l'audio, le stocker dans:
        audio_data = audio depuis le commencement de parler
        block_data = audio toute les 0.5 seconde pour avoir des infos de temps réel
        """
        self.audio_data = np.concatenate((self.audio_data, indata), axis=0)

        if time.time() - self.block_data_time > 0.5:
            self.block_data = np.empty((0, 1), dtype=np.float32)
            self.block_data_time = time.time()

        self.block_data = np.concatenate((self.block_data, indata), axis=0)


    def is_speaking(self):
        """
        Detecte si l'utilisateur parle par niveau sonore moyen
        """
        if len(self.block_data) == 0:
            return False
        
        if np.mean(self.block_data ** 2) > self.silence_threshold:
            return True
        else:
            return False


    def record_audio(self):
        """
        Record l'audio jusqu'à temps que l'utilisateur ne parle plus
        Retourne l'audio
        """
        self.audio_data = self.audio_data[-STT_Samplerate:]
        last_text = 0

        while time.time() - last_text < self.silence_timeout or last_text == 0:
            time.sleep(self.tick_time)
            if self.is_speaking():
                last_text = 0
            elif last_text == 0:
                last_text = time.time()

        self.is_recording = False
        return self.audio_data


    def transcribe(self, audio):
        """
        Transcrit l'audio en texte en français
        """
        start_transcribe_time = time.time()
        segments, _ = self.model.transcribe(audio.flatten(), language="fr", beam_size=10)
        text = "".join([seg.text for seg in segments]).strip()
        print(f"Time to transcribe : {time.time() - start_transcribe_time}")
        return text



    def run(self):
        """
        Boucle principale qui écoute l'utilisateur et affiche le texte reconnu
        """
        print("STT Initialiser")

        def process_audio():
            """
            Fonction pour gérer l'enregistrement et la transcription dans un thread séparé
            """
            text = self.transcribe(audio)
            print(f"🗣️{text}")
            self.new_prompt(text)
            self.is_recording = False


        with sd.InputStream(samplerate=STT_Samplerate, channels=1, callback=self.audio_callback):
            while True:
                if self.is_recording:
                    self.stop_generation()
                    audio = self.record_audio()
                    threading.Thread(target=process_audio).start() 
                    self.is_recording = False
                    print("FIN Parole détecter")
    
                else:
                    if self.is_speaking():
                        print("🎤 Parole détecter")
                        self.is_recording = True

                time.sleep(self.tick_time)


if __name__ == "__main__":
    transcriber = SpeechTranscriber()
    transcriber.run()
