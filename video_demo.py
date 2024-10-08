"""
Demo 1: Video demo

This demo shows how to use DINOStream with a video file. The input stream can be a video file or a camera index.
Running this demo should work on the example video, or you can pass in an integer ID to connect your webcam (ex: 0).
"""
import sys
from dino_stream import DINOStream

if __name__ == '__main__':

    default_video = 'assets/immature_stag.mp4'
    input_stream = default_video
    if len(sys.argv) > 1:
        input_stream = sys.argv[1]

    # Just for fun if the input stream is the default video, set a customized prompt
    default_prompt = "What is this small animal with antlers in front of me." if input_stream == default_video else ""
    try:
        dst = DINOStream(input_stream,
                         llm_model_id='dino_llama',  # Choosing this LLM, assuming Ollama is running
                         default_prompt=default_prompt,
                         )
        dst.start()
    except KeyboardInterrupt:
        pass