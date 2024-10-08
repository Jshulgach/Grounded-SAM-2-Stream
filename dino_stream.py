import os
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
from glob import glob

class DINOStream:
    """ Tracker class using Grounding DINO with LLM agent to perform contextualized object tracking in realtime.

    Parameters:
    =================
        input_stream: str, int, list             The input stream to be used for object detection. Can be a video file, camera index, or list of video files/camera indices.
        img_size: tuple (default: (1080, 720))   The size of the image to be used for processing.
        dino_model_id: str                       The model ID for the DINO model to be used for grounding.
        llm_model_id: str (default: "llama3.1")  The model ID for the LLM model to be used for generating responses.
        default_prompt: str (default: "")        The default prompt to be used for the LLM model.
        host: str (default: 'localhost')         The host address for the application server.
        port: int (default: 15555)               The port number for the application server.
        save_video: bool (default: False)        Whether to save the video output.
        image_directory: str (default: None)     The directory to save the images.
        save_interval: int (default: None)       The interval at which to save the images.

    """
    def __init__(self, input_stream,
                 img_size=(1080, 720),
                 dino_model_id="IDEA-Research/grounding-dino-tiny",
                 llm_model_id="llama3.1",
                 default_prompt="",
                 host='127.0.0.1',
                 port=15555,
                 save_video=False,
                 image_directory=None,
                 save_interval=None):
        self.img_size = img_size

        # Initialize video capture for single video/stream or list
        self.caps = []

        if not isinstance(input_stream, list): # Ensure it's a list, even if it's a single element
            input_stream = [input_stream]  # Convert single input to list for uniform processing
        for stream in input_stream:
            try:
                stream = int(stream)  # Convert to integer if it's a camera index (e.g., '0')
            except ValueError:
                pass  # Otherwise, keep it as a string (file path)

            # Create VideoCapture object and append to self.caps
            cap = cv2.VideoCapture(stream)
            if not cap.isOpened():
                print(f"Error: Could not open video stream {stream}")
            self.caps.append(cap)

        self.prompt = default_prompt
        self.host = host
        self.port = port
        self.save_video = save_video
        self.image_directory = image_directory
        self.save_interval = save_interval

        if self.save_video:
            self.out = cv2.VideoWriter('detection.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, self.img_size)

        self.query_queue = collections.deque([])
        self.response_queue = collections.deque([])
        self.all_stop = False
        self.last_save_time = [time.time()] * len(self.caps)
        self.frame_counter = 0

        self.query_task = None
        self.search_task = None

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
        if default_prompt != "":
            print("Default response: {}".format(default_prompt))
            self.query_queue.append(default_prompt)

    def start(self):
        """ Make async library call to run the main function """
        asyncio.run(self.main())

    async def main(self):
        """ Start main tasks and coroutines"""
        print("Setting up server on {}:{}".format(self.host, self.port))
        asyncio.create_task(asyncio.start_server(self.handle_client, self.host, self.port))

        # Run coroutine to begin streaming video
        asyncio.create_task(self.run_streaming())

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
            outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.4, target_sizes=[image.size[::-1]]
        )

        # Get box prompts for SAM2 (TO-DO)
        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        #class_ids = np.array(list(range(len(class_names))))
        class_ids = list(range(len(class_names)))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        #detections = sv.Detections(
        #    xyxy=input_boxes,  # (n, 4)
        #    mask=None, # masks.astype(bool),  # (n, h, w)
        #    class_id=class_ids
        #)

        #return results, labels, detections
        return results, labels, input_boxes, class_ids

    def annotate_frame(self, frame, labels, boxes):
        """Annotate the frame with detection results"""
        for i, box in enumerate(boxes):
            label = labels[i]
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def annotate_frame_old(self, frame, labels, detections):
        # Annotate the frame with detection results
        box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.DEFAULT)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return annotated_frame

    def save_frame(self, camera_id, frame):
        """ Saves the current frame to the specified image directory """
        if self.image_directory is not None:
            try:
                if not os.path.exists(self.image_directory):
                    os.makedirs(self.image_directory)
                cam_path = self.image_directory + f"/camera_{camera_id}"
                if not os.path.exists(cam_path):
                    os.makedirs(cam_path)

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(cam_path, f"frame_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved frame: {filename}")

            except Exception as e:
                print(f"Error saving frame: {e}")

    def get_all_jpg_files(self, directory):
        """Get all .jpg files in the given directory and its subdirectories."""
        if directory is None:
            print("Error: image_directory is not set.")
            return []  # Return an empty list if directory is None

        jpg_files = []
        for root, dirs, files in os.walk(directory):  # Walk through all subdirectories
            for file in files:
                if file.lower().endswith('.jpg'):  # Check if file ends with .jpg
                    jpg_files.append(os.path.join(root, file))  # Get full path of the file
        return jpg_files

    async def search_in_images(self, prompt):
        """Search for the prompted object in saved images"""
        latest_times = {}
        latest_files = {}

        print("Searching in saved images...")
        image_files = self.get_all_jpg_files(self.image_directory)  # Get all jpg files in the image directory
        for image_file in image_files:
            # Extract the timestamp from the file name
            camera_id = image_file.split('_')[1]  # Extract camera index from file name
            timestamp_str = image_file.split('_')[-1].split('.')[0]  # Assuming format: frame_camX_YYYYMMDD-HHMMSS.jpg
            timestamp = time.strptime(timestamp_str, "%Y%m%d-%H%M%S")

            image = Image.open(image_file)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.40, text_threshold=0.40, target_sizes=[image.size[::-1]]
            )

            if len(results[0]["boxes"]) > 0:  # Object found
                if camera_id not in latest_times or timestamp > latest_times[camera_id]:
                    latest_times[camera_id] = timestamp
                    latest_files[camera_id] = image_file

        for camera_id, latest_file in latest_files.items():
            print(f"Object last detected with camera {camera_id} at {time.strftime('%Y-%m-%d %H:%M:%S', latest_times[camera_id])}")

            # Convert the PIL image to OpenCV format and display it
            image = Image.open(latest_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert from RGB (PIL) to BGR (OpenCV)
            results, labels, boxes, class_ids = self.process_frame(image_cv, self.prompt)
            annotated_image = self.annotate_frame(image_cv, labels, boxes)

            # Display the detected image
            cv2.imshow(f"Detected Object - Camera {camera_id}", annotated_image)
            printf("Press q or Esc to close the window.")
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                pass

        if not latest_files:
            print("Object not found in saved images.")
        self.search_task = None  # Reset search task after completion
        self.prompt = ""  # Reset prompt

    async def run_streaming(self):
        """ Run the grounded tracking on the input stream   """
        print("Running grounded tracking with DINO. Press Esc or 'q' to quit.")

        # Check if camera is opened
        for cap in self.caps:
            if not cap.isOpened():
                print("Error: Could not open video stream")
                return

        while not self.all_stop:

            all_empty_boxes = True

            # Query handling can be independent of the camera streaming,but the query is applied to all cameras.
            # Check if there are queries to process. If so, create a background task to handle the query
            if self.query_queue and self.query_task is None:
                query = self.query_queue.popleft()
                self.query_task = asyncio.create_task(self.extract_handler(query, self.response_queue, self.llm_model))

            # Check if background task for query processing is done
            if self.query_task is not None and self.query_task.done():
                if self.response_queue:
                    self.prompt = self.response_queue.popleft()
                    print(f"Response: {self.prompt}")
                self.query_task = None  # Reset query task

            # Read frames from each camera
            for i, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to desired size
                frame = cv2.resize(frame, self.img_size)

                # Save frame at specified interval
                if self.save_interval and (time.time() - self.last_save_time[i]) > self.save_interval:
                    self.last_save_time[i] = time.time()
                    self.save_frame(i, frame)

                # Process each frame
                #results, labels, detections = self.process_frame(frame, self.prompt)
                results, labels, boxes, class_ids = self.process_frame(frame, self.prompt)

                if len(boxes) > 0:
                    # annotated_frame = self.annotate_frame(frame, labels, detections)
                    annotated_frame = self.annotate_frame(frame, labels, boxes)
                    all_empty_boxes = False # Object found in at least one camera
                else:
                    annotated_frame = frame

                cv2.imshow(f"Grounded Tracking (Camera {i})", annotated_frame)

                if self.save_video:
                    self.out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                #if cv2.waitKey(0) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                #if cv2.waitKey(0) & 0xFF == 27:  # Esc key
                break

            # If no object is found in any of the cameras and no search task is running, create a new search task
            if all_empty_boxes and self.search_task is None and self.prompt != "":
                print("Object not found in any camera. Searching in saved images.")
                self.search_task = asyncio.create_task(self.search_in_images(self.prompt))
            if self.search_task and self.search_task.done():
                self.search_task = None

            # Ensure tasks are running
            await asyncio.sleep(0)

        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()
        self.all_stop = True
