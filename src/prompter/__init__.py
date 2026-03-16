from .BatchProcessingChatPrompter import BachtProcessingChatPrompter
from .ChatPrompter import ChatPrompter
from .GenerationPrompter import GenerationPrompter

prompters = {
    "Chat": ChatPrompter,
    "Generation": GenerationPrompter,
    "BatchProcessingChat": BachtProcessingChatPrompter
}

def get_prompter(prompt_type, samplers):
    """
    Get the prompter for the given prompt type.
    """
    return prompters[prompt_type](samplers)