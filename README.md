# Fire, Smoke, and Leak Detection 🔥💨💦

This project uses a machine learning model to detect fire, smoke, and leaks from a webcam feed. The model was trained using Roboflow and is deployed on an NVIDIA Jetson Orin Nano.

## Setup

### Prerequisites

- NVIDIA Jetson Orin Nano
- JetPack 4.5, 4.6, or 5.1 installed on the Jetson
- Docker installed on the Jetson
- Python 3 installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/roboflow-fire-leak-detection.git
   cd roboflow-fire-leak-detection
   ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure that your Jetson is flashed with Jetpack 4.5, 4.6, or 5.1. You can check your existing setup with this repository from Jetson Hacks:
   ```bash
   git clone https://github.com/jetsonhacks/jetsonUtilities.git
   cd jetsonUtilities
   python jetsonInfo.py
   ```

4. Run the Roboflow Inference Server Docker container on the Jetson:
    ```bash
    sudo docker run --privileged --net=host --runtime=nvidia \
    --mount source=roboflow,target=/tmp/cache -e NUM_WORKERS=1 \
    roboflow/roboflow-inference-server-jetson-4.5.0:latest
    ```
    Note: The Docker image you need depends on what JetPack version you are using.

5. Ensure the captured directory exists:
    ```bash
    mkdir -p captured
    ```

### Configuration
Set your Roboflow API key and model ID in the `deploy_jetson.py` file:
    ```python
    # Set your API key and model ID
    API_KEY = "your_roboflow_api_key"
    MODEL_ID = "fire-smoke-leak-detection/1"
    ```

### Running the Application
To start the detection, run:
    ```bash
    python deploy_jetson.py
    ```
    The script will capture the webcam feed, perform inference using the Roboflow model, and save images with detected hazards to the captured folder.

### Local Testing
To test the application locally on your MacBook, use the deploy_local.py script. Set your Roboflow API key and model ID, and run:
    ```bash
    python deploy_local.py
    ```

### Directory Structure
    ```
    roboflow-fire-leak-detection/
    ├── captured/
    ├── deploy_local.py
    ├── deploy_jetson.py
    ├── README.md
    ├── requirements.txt
    └── .gitignore
    ```

### License
This project is licensed under the MIT License. See the LICENSE file for details.
