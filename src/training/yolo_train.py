import os
from ultralytics import YOLO
from pathlib import Path

import argparse

### SETTINGS ###

RUN_SIGNATURE = 'model_imgsz_args'

DATA_YAML_FORMAT = 'data_X.yaml'
DATA_YAML_PATH = Path("/home/diaz_asian/StampDetector/data/YOLO_CONFIGS")
MODEL = "yolo11n.pt"
IMG_SIZE = 640
EPOCHS = 300
BATCH = -1                  # auto-batch 
NAME = f"v11n_640_default"
PROJECT = "StampDetector"

DEFAULT_ARGS = {
    "batch": BATCH,
    "imgsz": IMG_SIZE,
    "project": PROJECT,
    "exist_ok": True,
    "device": 0,
    "workers": 4,
    "pretrained": True,
    "patience": 100
}

CUSTOM_ARGS = {
    "batch": BATCH,
    "imgsz": IMG_SIZE,
    "project": PROJECT,
    "exist_ok": True,
    "device": 0, # GPU
    "workers": 4,
    "pretrained": True,

    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,

    "box": 10.0,      # heavily penalize bad coordinates
    "cls": 0.0,       # completely ignore classification
    "dfl": 2.0,

    "patience": 100,   # Early stopping

    "rect": False,           
    "close_mosaic": 15,           # last 15 epochs without mosaic → clean boxes
    "cache": "disk",
    "amp": True,
    "augment": True,              # keeps test-time augmentation
    "plots": True,
    "save": True,
}

def main():

    parser = argparse.ArgumentParser(description="Run YOLO training")

    parser.add_argument("--n_epochs", type=int, required=True,
                    help="number of epochs", default=EPOCHS)
    
    parser.add_argument("--im_size", type=int, required=True,
                    help="Image size", default=IMG_SIZE, choices=[640,800,1024])
    
    parser.add_argument("--model", type=str, required=True,
                help="model to use ", default=MODEL, choices=[''
                'yolo11n.pt',
                'yolo11s.pt',
                'yolov8n.pt',
                'yolov8s.pt'])
    
    parser.add_argument("--h_params", type=str, required=True,
                help="which hyper parameter set to use", default='default', choices=['default', 'custom'])

    args = parser.parse_args()

    im_size = args.im_size
    data_yaml = DATA_YAML_PATH / DATA_YAML_FORMAT.replace('X',str(im_size))
    h_params = args.h_params

    train_args = None
    if h_params == 'default':
        train_args = DEFAULT_ARGS
    elif h_params == 'custom':
        train_args = CUSTOM_ARGS

    model = args.model
    
    name = RUN_SIGNATURE.replace('model',model.split('.')[0]).replace('imgsz',str(im_size)).replace('args',h_params)
    
    train_args['data'] = data_yaml
    train_args['imgsz'] = im_size
    train_args['epochs'] = args.n_epochs
    train_args['name'] = name

    ### clearn RAM / GPU memory before training
    print('cleaning RAM / GPU memory')
    os.system('bash clean_memory.sh')
    
    ### TRAINING ###
    model = YOLO(model)

    print(f"Training {model} on {Path(data_yaml).parent} for {args.n_epochs} epochs")
    print(f"Results will be saved to: runs/detect/{PROJECT}/{NAME}")

    results = model.train(**train_args)
        
if __name__ == "__main__":
    main()