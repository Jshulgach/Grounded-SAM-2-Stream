import ollama

class LlamaWrapper:
    """ Wrapper class for the Ollama API."""
    def __init__(self, model_id='llama3.1'):
        self.model_id = model_id

        # Check if the model exists, if so say it is up and running
        models = self.list_models()
        if not self.model_exists(model_id, models):
            raise ValueError(f"Model {model_id} not found in Ollama models.")
        else:
            print(f"Model {model_id} is up and running.")

    def list_models(self):
        """ Method to list available models from Ollama """
        # This would be the call to get the list of models, as you already did
        return ollama.list()['models']

    def model_exists(self, model_id, available_models):
        """Check if the requested model exists in the available models."""
        for model in available_models:
            if model_id in model['name'] or model_id in model['model']:
                return True
        return False

    def send(self, message):
        """ Sends a blocking message to the Ollama API and returns the response."""
        response = ollama.chat(model=self.model_id, messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        return response['message']['content']

