import cv2
from pathlib import Path
import numpy as np
from typing import Tuple, List

COLOR_STAMP = (200, 0, 0)    # BGR – blue
COLOR_TEXT  = (0, 0, 200)    # red
FONT = cv2.FONT_HERSHEY_SIMPLEX

def resize_and_pad(img: np.ndarray, target_size:int) -> Tuple[np.ndarray, float, Tuple[int, int]]:

    """
    Resize image to fit within target_size × target_size while preserving aspect ratio,
    then pad with white borders (255) to exactly target_size × target_size.
    
    Args:
        img: Input image as uint8 numpy array (H, W) or (H, W, C)
        target_size: Desired output square size (both width and height)
    
    Returns:
        padded: Final image of shape (target_size, target_size, C) with white padding
        scale: Scaling factor applied (target_size / max(original_h, original_w))
        (left, top): Padding applied on left and top sides
    """

    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=255)  # white padding
    return padded, scale, (left, top)

def predict_and_draw(original_img: np.ndarray, results:List, target_size:int, thr:float) -> np.ndarray:

    """
        Draw detection boxes on the original image by reversing the resize+pad transformation
        applied during inference (e.g., 640×640 letterboxed input).

        Args:
            original_img: Original input image (H, W, 3) as uint8 np.ndarray
            results: List of ultralytics YOLO Results objects (usually length 1 for single image)
            target_size: The square size used during preprocessing (default 640)
            thr: Confidence threshold for displaying detections

        Returns:
            np.ndarray: Copy of original_img with drawn boxes and labels
    """

    img = original_img.copy()
    h, w = original_img.shape[:2]

    for r in results:
        boxes = r.boxes.cpu().numpy()
        for box in boxes:
            if box.conf[0] < thr:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Convert from padded 640×640 back to original size
            scale = min(target_size / w, target_size / h)
            pad_l = (target_size - w * scale) // 2
            pad_t = (target_size - h * scale) // 2

            x1 = int((x1 - pad_l) / scale)
            y1 = int((y1 - pad_t) / scale)
            x2 = int((x2 - pad_l) / scale)
            y2 = int((y2 - pad_t) / scale)

            conf = box.conf[0]

            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_STAMP, 3)
            label = f"Stamp {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 1.1, 3)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), COLOR_TEXT, -1)
            cv2.putText(img, label, (x1, y1 - 5), FONT, 1.1, (255, 255, 255), 3)

    return img