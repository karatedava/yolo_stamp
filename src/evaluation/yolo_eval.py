from pathlib import Path
import os
from tqdm import tqdm
from ultralytics import YOLO
from preprocessing import yolop
import cv2
import numpy as np
import pandas as pd

CONF=0.5
IOU=0.5
MODEL_PATH = Path("../trained_models")

def main():

    # ============== OUR DATA ============== #
    no_stamps_path = Path(f'../../data/raw/unlabeled/dataset/bg')
    stamped_path = Path(f'../../data/raw/unlabeled/dataset/stamps')

    model_names = os.listdir(MODEL_PATH)

    #cols = ['model','test_count','test_match','test_mAP05','false_alarm','true_alarm']
    results = []
    for model_name in model_names:

        print(f'evaluating: {model_name}')

        model = YOLO(MODEL_PATH / model_name)
        imsz = int(model_name.split('_')[2])

        # ============== TEST SET ============== #

        test_images_path = Path(f'../../data/processed_{imsz}/images/test')
        test_labels_path = Path(f'../../data/processed_{imsz}/labels/test')
        test_set_yaml_path = Path(f'../../data/YOLO_CONFIGS/test_{imsz}.yaml')

        mae, exact_matches, box_map50, box_map = test_set_eval(
            model=model,
            test_imgs_path=test_images_path,
            test_labels_path=test_labels_path,
            test_set_yaml=test_set_yaml_path
        )

        false_alarms, true_alarms = our_data_eval(
            model=model,
            no_stamps_path=no_stamps_path,
            stamped_path=stamped_path,
            imsz=imsz
        )

        results.append({
                'model': model_name,
                'test_mae': mae,
                'test_match': exact_matches,
                'test_mAP05': box_map50,
                'false_alarm': false_alarms,
                'true_alarm': true_alarms
            })
        
    df = pd.DataFrame(results, columns=results[0].keys())
    df.to_csv('eval_data/evaluation_run.csv', index=False)
        
def test_set_eval(model:YOLO, test_imgs_path:Path, test_labels_path:Path, test_set_yaml:Path):

    """
    Return MAE, exact_match, mAP05, mAP05_95
    """

    ### STAMP COUNTING EVALUATION ###
    
    images = [f for f in os.listdir(test_imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    count_abs_errors = [] # predicted stamps - actual stamps
    exact_matches = 0 # detected number of stamps == actual number of stamps

    for img_name in tqdm(images):
        img_path = os.path.join(test_imgs_path, img_name)
        label_path = os.path.join(test_labels_path, os.path.splitext(img_name)[0] + '.txt')

        with open(label_path) as f:
            stamp_count = len([line for line in f if line.strip()])
        
        results = model(img_path, conf=CONF, iou=IOU, verbose=False)[0]
        pred_count = len(results.boxes)
        count_abs_errors.append(abs(pred_count - stamp_count))

        if pred_count == stamp_count:
            exact_matches += 1

    mae = np.mean(count_abs_errors)
    exact_pct = exact_matches / len(images) * 100

    print("\n=== Counting Performance ===")
    print(f"Mean Absolute Error (count)   : {mae:.4f}")
    print(f"Exact count correct        : {exact_pct:.1f}% ({exact_matches}/{len(images)}) images)")

    ### BOUNDING BOX REGRESSION EVALUATION ###

    metrics = model.val(
        data       = test_set_yaml,
        conf       = CONF,
        iou        = IOU,
        cls        = 0.001,      # ignore classification
        plots      = False,
        save_json  = False,
    )

    box_map50 = metrics.box.map50
    box_map = metrics.box.map

    print(f"Box mAP@0.5     : {box_map50:.4f}")
    print(f"Box mAP@0.5:0.95: {box_map:.4f}")

    return mae, exact_matches, box_map50, box_map

def our_data_eval(model:YOLO, no_stamps_path:Path, stamped_path:Path, imsz:int):
    
    """
    returns: false_alarms, true_alarms
    """

    false_alarms = 0
    images_no_stamp = os.listdir(no_stamps_path)
    for img_file in images_no_stamp:
        img = cv2.imread(str(no_stamps_path / img_file))

        img_processed, _, _ = yolop.resize_and_pad(img,imsz)
        results = model(img_processed, conf=CONF, iou=IOU, verbose=False)[0]
        pred_count = len(results.boxes)
        
        if pred_count != 0:
            false_alarms += 1
    
    false_alarms /= len(images_no_stamp)
    
    true_alarms = 0
    images_stamped = os.listdir(stamped_path)
    for img_file in images_stamped:
        img = cv2.imread(str(stamped_path / img_file))

        img_processed, _, _ = yolop.resize_and_pad(img,imsz)
        results = model(img_processed, conf=CONF, iou=IOU, verbose=False)[0]
        pred_count = len(results.boxes)
        print(pred_count)

        if pred_count > 0:
            true_alarms += 1
    
    true_alarms /= len(images_stamped)

    return false_alarms, true_alarms

if __name__ == "__main__":
    main()