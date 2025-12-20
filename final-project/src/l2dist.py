import numpy as np
from PIL import Image

def l2_similarity(base_img: Image.Image, img: Image.Image) -> float:
    """
    Crop non-background regions from img1 and img2 (if bg is None, prefer transparency as background),
    resize img1 or img2 size, and return the similarity ratio [0, 1] between the two images' pixel vectors.
    """

    base_img = base_img.convert("RGBA")
    img = img.convert("RGBA").resize(base_img.size)

    # Compute L2 on flattened arrays
    arr1 = np.asarray(base_img, dtype=np.float32).ravel()
    arr2 = np.asarray(img, dtype=np.float32).ravel()

    l2_distance = np.linalg.norm(arr1 - arr2, ord=2)
    max_distance = np.sqrt(len(arr1)) * 255  # Maximum possible distance

    similarity = 1 - (l2_distance / max_distance)
    return float(np.clip(similarity, 0, 1)) # ensure the value is in [0, 1]
