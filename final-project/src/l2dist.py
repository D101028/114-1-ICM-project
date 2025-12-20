import numpy as np
from PIL import Image

def l2_similarity(base_img: Image.Image, img: Image.Image) -> float:
    
    """
    Compute the l2 distance of two images and normalize into [0, 1]. 

    Resize `img` based on the size of `base_img`, and return the similarity 
    ratio [0, 1] between the two images' pixel vectors.

    :param base_img: The first, also the based image. 
    :type base_img: Image.Image
    :param img: The second image. 
    :type img: Image.Image
    :return: A normalized value in [0, 1] as the similarity. 
    :rtype: float
    """

    base_img = base_img.convert("L")
    img = img.convert("L").resize(base_img.size)

    # Compute L2 on flattened arrays
    arr1 = np.asarray(base_img, dtype=np.float32).ravel()
    arr2 = np.asarray(img, dtype=np.float32).ravel()

    l2_distance = np.linalg.norm(arr1 - arr2, ord=2)
    max_distance = np.sqrt(len(arr1)) * 255  # Maximum possible distance

    similarity = 1 - (l2_distance / max_distance)
    return float(np.clip(similarity, 0, 1)) # ensure the value is in [0, 1]
