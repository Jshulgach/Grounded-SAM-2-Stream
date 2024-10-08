"""
Demo 2: Multi-camera demo

This demo shows how to use DINOStream with multiple cameras. The input stream can be a list of camera indices or a list of video files.
To run this demo, you need to have multiple cameras connected to your computer. You can also use video files as input streams.
"""

from dino_stream import DINOStream

if __name__ == '__main__':

    # Setting up parameters
    input_stream = [0, 1]  # Multiple streams can be provided as a list of camera indices or video files
    image_root_dir = 'images'  # Directory to save images (either absolute or relative path)

    try:
        dst = DINOStream(input_stream,
                         llm_model_id='dino_llama',  # Choosing this LLM, assuming Ollama is running
                         image_directory=image_root_dir,
                         save_interval=60,  # Save images every 60 seconds
                         )
        dst.start()

    except KeyboardInterrupt:
        pass
