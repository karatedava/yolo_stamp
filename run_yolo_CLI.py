import argparse
import json
from ultralytics import YOLO
import cv2
from src.preprocessing import yolop
from pathlib import Path

# ============== CONFIG ==============
MODEL_PATH = Path("src/trained_models")
INPUT_SIZE = 640

CONF=0.5
IOU=0.5

COLOR_STAMP = (200, 0, 0)    # BGR – blue
COLOR_TEXT  = (0, 0, 200)    # red
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="Run YOLO stamp detection on a single image")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input image (e.g. image.jpg or image.png)")
    parser.add_argument("--output", "-o", type=str, default="predictions",
                        help="Output directory where the result will be saved (default: ./predictions)")
    parser.add_argument("--model", "-m", type=str, default="best_v11n_640_default_1000e.pt",
                    help="trained yolo model to use")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Validate input
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {MODEL_PATH} ...")
    model_path = MODEL_PATH / args.model
    model = YOLO(model_path)

    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Error: Could not read image: {input_path}")
        return 1

    orig_h, orig_w = img.shape[:2]

    print(f"Processing {input_path.name} ...")

    # 1. Preprocess exactly like during training
    img_padded, _, _ = yolop.resize_and_pad(img, INPUT_SIZE)

    # 2. Inference
    results = model(img_padded, conf=CONF, iou=IOU, verbose=False)[0]

    # 3. Draw on original image
    final = yolop.predict_and_draw(img, [results], INPUT_SIZE, CONF)  # wrapped in list because function expects iterable

    # 4. Save result
    out_path = output_dir / input_path.name
    cv2.imwrite(str(out_path), final)
    print(f"Saved prediction: {out_path}")

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        # Convert from xyxy (tensor) to list of floats
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "class": results.names[cls_id],   # e.g. "stamp"
                "confidence": float(conf),
                "bbox": {
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2))
                }
            })

    json_data = {
        "image_width": orig_w,
        "image_height": orig_h,
        "detections": detections
    }
    json_path = output_dir / (input_path.stem + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Detection JSON saved: {json_path}")
    print(f"   Found {len(detections)} stamp(s)")

    return 0


if __name__ == "__main__":
    main()