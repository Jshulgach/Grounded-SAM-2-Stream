# Grounded SAM 2
Grounded tracking on anything in realtime using natural language queries with [Grounding DINO](https://arxiv.org/abs/2303.05499), [Grounding DINO 1.5](https://arxiv.org/abs/2405.10300), and [Ollama](https://github.com/ollama/ollama) which provides easy access to various [LLMs](https://arxiv.org/abs/2307.06435). 


<div align=center>

<p align="center">
<img src="./assets/dino-stream.gif" width="720">
</p>

</div>


## Contents
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Single Video Demos](#video-demo)
- [Multi-camera streaming](#multi-camera-streaming-demo)
- [Acknowledgements](#acknowledgements)

## Pre-requisites
- [Git](https://git-scm.com/downloads)
- [Anaconda](https://www.anaconda.com/products/distribution)
- [Python 3.10](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [Nvidia CUDA Toolkit (optional)](https://developer.nvidia.com/cuda-toolkit)

## Installation

1. Make sure you have Anaconda installed. If not, you can download it [here](https://www.anaconda.com/products/distribution).

2. Prepare your environment:
   - Create the conda environment with python 3.10
   ```bash
   conda create -n dino-llama python=3.10 -y
   conda activate dino-llama
   ```
    - Install [pytorch](https://pytorch.org/) for your system. Follow the official documentation for your specific system.

    - Install the rest of the dependencies from the package
    ```
    git clone https://github.com/Jshulgach/Grounded-SAM-2-Stream.git
    cd Grounded-SAM-2-Stream 
    pip install -e .
    ```
<!--
#2. Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```
- (Optional) 'Grounding DINO' checkpoints should be automatically installed through first-time usage of the model within `transformers` but if you want to install them locally.
     ```bash
     mkdir gdino_checkpoints
     cd gdino_checkpoints
     bash download_ckpts.sh
    ```
-->
3. Prepare the LLM model:
    - Install Ollama on your respective OS (Linux, MacOS, Windows) by following the [readme instructions](https://github.com/ollama/ollama)
    - Create the llm model with the customized prompt.
    ```bash
    ollama create dino_llama -f <\path\to\repo>\llm\Modelfile
    ```


4. Set up CUDA for GPU usage with Grounding DINO (Optional). You should be able to download the CUDA Toolkit from the [Nvidia website](https://developer.nvidia.com/cuda-toolkit).  
<!--
  -  Since we need the CUDA compilation environment to compile the `Deformable Attention` operator used in Grounding DINO, we need to check whether the CUDA environment variables have been set correctly (which you can refer to [Grounding DINO Installation](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) for more details). You can set the environment variable manually as follows if you want to build a local GPU environment for Grounding DINO to run Grounded SAM 2:

  - ```bash
    export CUDA_HOME=/path/to/cuda-12.1/
    ```
-->
## Video Demo

A demo is included to handlea single video directory passed (or number for webcam). The DINOStream application creates a server and listens for messages from a client. 
```
python dino_stream.py 0 
```

Open a separate terminal and send a message to the server. The default server address is `localhost` and the default port is `15555`. You can change the server address and port in the `dino_stream.py` file:
```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 15555))
s.send(b'I am looking for something blue that squirts water')
```

## Multi-Camera Streaming Demo

To run the multi-camera streaming demo, you can run the following command which opens multiple webcams (assuming you have more than one connected):
```
python multi_stream.py
```


<!--
### Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTex
@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos}, 
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714}, 
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounding,
      title={Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection}, 
      author={Tianhe Ren and Qing Jiang and Shilong Liu and Zhaoyang Zeng and Wenlong Liu and Han Gao and Hongjie Huang and Zhengyu Ma and Xiaoke Jiang and Yihao Chen and Yuda Xiong and Hao Zhang and Feng Li and Peijun Tang and Kent Yu and Lei Zhang},
      year={2024},
      eprint={2405.10300},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@misc{jiang2024trex2,
      title={T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy}, 
      author={Qing Jiang and Feng Li and Zhaoyang Zeng and Tianhe Ren and Shilong Liu and Lei Zhang},
      year={2024},
      eprint={2403.14610},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
-->

### Acknowledgements
<!-- - [Segment-Anything-2-Real-Time](https://github.com/Gy920/segment-anything-2-real-time) -->
<!-- - [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) -->
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- - [Ollama](https://ollama.com/)

