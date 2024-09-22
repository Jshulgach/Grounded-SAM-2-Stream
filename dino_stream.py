import os
import sys
import cv2
import time
import asyncio
import torch
import numpy as np
from PIL import Image
import collections
import supervision as sv
from supervision.draw.color import ColorPalette
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from llm.ollama import LlamaWrapper


class DINOStream:
    """ Tracker class using Grounding DINO with LLM agent to perform contextualized object tracking in realtime.
    """
    def __init__(self, input_stream,
                 img_size=(1080, 720),
                 dino_model_id="IDEA-Research/grounding-dino-tiny",
                 llm_model_id="llama3.1",
                 default_prompt="",
                 host='127.0.0.1',
                 port=15555):
        self.cap = cv2.VideoCapture(input_stream) # cv2.VideoCapture(0) # camera
        self.img_size = img_size
        self.prompt = default_prompt
        self.host = host
        self.port = port
        self.query_queue = collections.deque([])
        self.response_queue = collections.deque([])
        self.all_stop = False

        # Set up torch environment
        # use bfloat16 for the entire notebook
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if self.device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        torch.inference_mode()

        # Set up Grounding DINO model
        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)

        # Set up LLM Model (maybe will allow other options, but for now will use the ollama service)
        self.llm_model = LlamaWrapper(model_id=llm_model_id)
        print("Model initialized")
        if self.prompt != "":
            print("Default response: {}".format(self.prompt))
            self.query_queue.append(self.prompt)

    def start(self):
        """ Make async library call to run the main function """
        asyncio.run(self.main())

    async def main(self):
        """ Start main tasks and coroutines"""

        # Start a server to listen for data from a port
        print("Setting up server on {}:{}".format(self.host, self.port))
        asyncio.create_task(asyncio.start_server(self.handle_client, self.host, self.port))

        # Run coroutine to begin streaming video
        asyncio.create_task(self.stream_video())

        while self.all_stop != True:
            await asyncio.sleep(0)  # Calling #async with sleep for 0 seconds allows coroutines to run

    async def handle_client(self, reader, writer):
        """ Callback function that gets called whenever a new client connection is established.
            Messages received will be 'queued' for handling.
        """
        client_addr = writer.get_extra_info('peername')
        print(f"Client {client_addr} connected")

        request = None
        while request != 'quit':
            try:
                request = (await reader.read(255)).decode('utf8')
                if not request:
                    break # Client disconnected

                print("Received: {}".format(request))
                self.query_queue.append(request)

            except Exception as e:
                print("Error: {}".format(e))
                break

        writer.close()
        await writer.wait_closed()

    async def extract_handler(self, query, queue, model):
        if query != "":
            self.new_query = True
        if query[-1] != ".":
            query += "."   # Make sure the query ends with a "."
        queue.append(self.llm_model.send(query))  # Waits for the response from the LLM model

    def process_frame(self, frame, prompt):
        """
        Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
        """
        # Convert frame to PIL image and apply Grounding DINO and SAM2
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.3, target_sizes=[image.size[::-1]]
        )

        # Get box prompts for SAM2 (TO-DO)
        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=None, # masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        return results, labels, detections

    def annotate_frame(self, frame, labels, detections):
        # Annotate the frame with detection results
        box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.DEFAULT)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return annotated_frame

    async def stream_video(self):
        """ Run the grounded tracking on the input stream   """
        print("Running grounded tracking with DINO. Press Esc or 'q' to quit.")
        query_task = None

        # Check if camera is opened
        if not self.cap.isOpened():
            print("Error: Could not open video stream")
            return

        while not self.all_stop:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize frame to desired size
            frame = cv2.resize(frame, self.img_size)

            if self.query_queue and query_task is None:
                query = self.query_queue.popleft()
                query_task = asyncio.create_task(self.extract_handler(query, self.response_queue, self.llm_model))

            # Check if background task for query processing is done
            if query_task is not None and query_task.done():
                if self.response_queue:
                    self.prompt = self.response_queue.popleft()
                    print("Response: {}".format(self.prompt))

                query_task = None # Reset query task

            # Process each frame
            results, labels, detections = self.process_frame(frame, self.prompt)

            # Annotate and display frame
            annotated_frame = self.annotate_frame(frame, labels, detections)
            cv2.imshow("Grounded Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Ensure tasks are running
            await asyncio.sleep(0)

        self.cap.release()
        cv2.destroyAllWindows()
        self.all_stop = True


if __name__ == '__main__':

    default_video = 'assets/immature_stag.mp4'
    input_stream = default_video
    if len(sys.argv) > 1:
        input_stream = sys.argv[1]

    # Just for fun if the input stream is the default video, set the default promot to be 'hands'
    default_prompt = "Is there something alive with antlers in front of me." if input_stream == default_video else ""
    try:
        dst = DINOStream(input_stream,
                         llm_model_id='dino_llama',  # Assuming the model is already created with Ollama
                         default_prompt=default_prompt,
                         )
        dst.start()
    except KeyboardInterrupt:
        pass
