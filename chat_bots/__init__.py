from .chat_with_gpt import GPTChatBot
def get_chat_model(model_name, api_key):
    cls = GPTChatBot
    
    if model_name is not None:
        return cls(api_key=api_key, model_name=model_name)
    else:
        return cls(api_key=api_key)