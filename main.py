import MLG
import stt



def stop_generation():
    MLG.MLG_queue_callback.stop = True


def new_prompt(text):

    print("Commence à générer le texte")
    MLG.reponse(MLG_model, text)


MLG_model = MLG.Load_MLG("C:/Users/lucas/OneDrive/Desktop/GlobalAI/Model/Meta-Llama-3-8B-Instruct-Q6_K.gguf")


transcriber = stt.SpeechTranscriber(new_prompt, stop_generation)
transcriber.run()