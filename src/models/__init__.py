def get_model(model_name):
    """All models in the study are served through vLLM via the Llama wrapper."""
    from .llama import Llama_Model
    return Llama_Model
