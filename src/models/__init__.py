def get_model(model_name):
    if model_name.startswith("meta-llama/Meta-Llama"):
        from .llama import Llama_Model
        return Llama_Model
    elif model_name.startswith("mistralai/Mistral"):
        from .Mistral import Mistral_Model
        return Mistral_Model
    elif model_name.startswith("mistralai/Mixtral"):
        from .Mixtral import Mixtral_Model
        return Mixtral_Model
    elif model_name.startswith("gpt"):
        from .ChatGPT import ChatGPT_API
        return ChatGPT_API
    elif model_name.startswith("mockgpt"):
        from .MockGPT import MockGPT
        return MockGPT
    else:
        from .llama import Llama_Model
        return Llama_Model