from typing import Tuple

import numpy as np
from PIL import Image

from .cluster import ClusterGroup

def l2_similarity(base_cluster: ClusterGroup, cluster: ClusterGroup) -> float:
    """
    Compute the l2 distance of two images and normalize into [0, 1]. 
    """
    base_img = base_cluster.to_L()
    img = cluster.to_L().resize(base_img.size)

    # Compute L2 on flattened arrays
    arr1 = np.asarray(base_img, dtype=np.float32).ravel()
    arr2 = np.asarray(img, dtype=np.float32).ravel()

    l2_distance = np.linalg.norm(arr1 - arr2, ord=2)
    max_distance = np.sqrt(len(arr1)) * 255  # Maximum possible distance

    similarity = 1 - (l2_distance / max_distance)
    return float(np.clip(similarity, 0, 1)) # ensure the value is in [0, 1]
