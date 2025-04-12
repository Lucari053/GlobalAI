from locallm import LocalLm, LmParams, InferenceParams
import llama_cpp


mistral_path = "mistral-7b-v0.1.Q4_K_M.gguf"
max_token = 8192
temperature = 0.7


mistral = LocalLm(
    LmParams(
        models_dir= "",
        is_verbose= False
        ))

mistral.load_model(mistral_path, max_token, gpu_layers= 999)

template = "[INST] Tu es un assistant francophone[/INST]"

print("GPU activé :", llama_cpp.llama_model_loader.LLAMA_CUDA)

"""

response = mistral.infer(
    "Quelles sont les planètes du système solaire",
    InferenceParams(
        template = None,
        temperature = temperature,
        max_tokens= 200,
        stream = False
    )
)

print(response)
"""