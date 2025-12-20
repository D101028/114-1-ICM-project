import numpy as np
from scipy.spatial import KDTree
from typing import Tuple

from .cluster import ClusterGroup

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def chamfer_ratio(base_cluster: ClusterGroup, cluster: ClusterGroup, topleft: Tuple[int, int]):
    # resize 
    _, _, h, w = base_cluster.get_bbox_yxhw()
    cluster = cluster.resize((w, h))

    arr1 = np.empty((0, 2), dtype=int)
    arr2 = np.empty((0, 2), dtype=int)

    for comp in base_cluster.components:
        pts = comp.pixels_below_threshold(127)   # shape (K,2)
        if pts.size == 0:
            continue
        arr1 = np.vstack([arr1, pts])

    for comp in cluster.components:
        pts = comp.pixels_below_threshold(127)
        pts[:, 0] = pts[:, 0] - topleft[0]
        pts[:, 1] = pts[:, 1] - topleft[1]
        if pts.size == 0:
            continue
        arr2 = np.vstack([arr2, pts])

    # 1. 處理空集合的情況 (避免 KDTree crash 或除以零)
    if arr1.size == 0 or arr2.size == 0:
        return 0.0 # 若一方為空，相似度為 0

    # 2. 計算 Chamfer Distance
    c_dist = chamfer_distance(arr1, arr2)

    # 3. 計算標準化基準 (對角線 D)
    # 建議使用兩者聯集的範圍，避免 ratio 變成負數 (因為 1 - ratio)
    all_pts = np.vstack([arr1, arr2])
    min_pt = all_pts.min(axis=0)
    max_pt = all_pts.max(axis=0)
    w_union = max_pt[1] - min_pt[1]
    h_union = max_pt[0] - min_pt[0]
    
    diagonal = (w_union**2 + h_union**2)**0.5
    
    if diagonal == 0:
        return 1.0 if c_dist == 0 else 0.0

    # 4. 標準化
    # 因為 chamfer_distance 回傳的是兩個 mean 的「和」
    # 所以最大值趨近於 2 * diagonal
    normalized_dist = c_dist / (2 * diagonal)
    
    # 限制在 [0, 1] 之間，防止浮點數運算誤差
    normalized_dist = min(1.0, normalized_dist)

    return 1 - normalized_dist
