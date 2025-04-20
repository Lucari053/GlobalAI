from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


class QueueCallback(BaseCallbackHandler):
    def __init__(self, stop):
        self.stop = stop

    def on_llm_new_token(self, token: str, **kwargs): 
        if self.stop:
            print("Interrompt la génération")
            raise KeyboardInterrupt("Generation manually stopped by control flag.")

        print(token, end='', flush=True)

    def on_llm_end(self, *args, **kwargs):
        print("\n[Generation finished]")

        
    


def Load_MLG(model_path):
    "Load text to text model"

    global MLG_queue_callback



    MLG_queue_callback = QueueCallback(stop = False)

    MLG_callback_manager = CallbackManager([MLG_queue_callback])

    model = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        streaming = True,
        n_ctx=2048,
        n_gpu_layers=-1,
        n_batch=512, 
        max_tokens=256,
        repeat_penalty=3,
        top_p=0.9,
        verbose=False,
        stop=["FIN", "[FIN]"],
        callback_manager = MLG_callback_manager
        )
    
    return model


def reponse(model, user_input):

    MLG_queue_callback.stop = False

    template = PromptTemplate.from_template(
        "Tu es un assistant intelligent qui répond en français de manière claire et concise.\n"
        "Termine toujours ta réponse par [FIN].\n\n"
        "Question : {question}\nRéponse :"
    )

    prompt = template.format(question=user_input)

    try:
        result = model(prompt)
    except KeyboardInterrupt:
        1+1

    


if __name__ == "__main__":
    
    MLG =  Load_MLG("C:/Users/lucas/OneDrive/Desktop/GlobalAI/Model/Meta-Llama-3-8B-Instruct-Q6_K.gguf")
    reponse(MLG, "Raconte moi l'histoire du Royaume-Unis")