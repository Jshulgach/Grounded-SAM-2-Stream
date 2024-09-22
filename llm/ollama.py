import ollama

class LlamaWrapper:
    """ Wrapper class for the Ollama API."""
    def __init__(self, model_id='llama3.1'):
        self.model_id = model_id
        print(f"Connecting to Ollama model {model_id}")

    def send(self, message):
        """ Sends a blocking message to the Ollama API and returns the response."""
        response = ollama.chat(model=self.model_id, messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        return response['message']['content']

