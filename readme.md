# DINOv3-YOLOS Detection

An experimental fork of the original NeurIPS 2021 YOLOS repository that adds support for loading DINOv3 ViT backbones from Hugging Face, enabling an evaluation of how DINOv3 features transfer to object detection when adapted to a YOLOS-style pipeline. The repository also includes several experiment-friendly utilities to facilitate running and reproducing experiments.

## Performance
- Dataset: PASCAL VOC 2012
- Training: 75 epochs
- Config: `configs/finetune.yaml`
- Metric: mAP@0.5 (validation)

Result: mAP@0.5 = 0.66

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
   Place  COCO-style data under `data/`.  
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
    