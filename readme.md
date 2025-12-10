# DINOv3-YOLOS Detection

A experimental repo for running YOLOS object detection with a DINOv3 ViT backbone.  
This is a lightweight fork of the original `hustvl/YOLOS` codebase, with added support for loading DINOv3 models from HuggingFace and running experiments on VOC.

## Run

1. Install dependencies  
    ```
    pip install -r requirements.txt
    ```

2. Log in to Hugging Face (required for loading the DINOv3 backbone)  
    ```
    huggingface-cli login
    ```

3. Prepare the dataset  
   Place VOC or COCO-style data under `data/`.  
   To convert VOC to COCO format:  
    ```
    sh voc2coco/voc2coco.sh
    ```

4. Select a configuration file  
   For example: `configs/freeze.yaml`.  
   Adjust dataset paths, backbone name, and other settings as needed.

5. Launch training or evaluation  
    ```
    torchrun main.py --config_path path/to/configuration/file.yaml
    ```
    
