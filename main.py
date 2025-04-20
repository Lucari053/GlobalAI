import MLG
import stt
import threading
import time




def stop_generation():
    MLG.MLG_queue_callback.stop = True


def new_prompt(text):

    print("Commence à générer le texte")
    MLG.reponse(MLG_model, text)


MLG_model = MLG.Load_MLG("")


transcriber = stt.SpeechTranscriber(new_prompt, stop_generation)
transcriber.run()