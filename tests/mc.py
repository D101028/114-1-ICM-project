import numpy as np
import cv2
import math
from PIL import Image
from scipy.ndimage import label, center_of_mass

def level_cluster_centroids(centroids, threshold, wx=1.0, wy=1.0):
    """
    一般 cluster： BFS + weighted distance。
    回傳一層 cluster 結果。
    """
    n = len(centroids)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster = [i]

        while queue:
            cur = queue.pop()
            cx, cy = centroids[cur]

            for j in range(n):
                if visited[j]:
                    continue

                x, y = centroids[j]

                dx = (cx - x) * wx
                dy = (cy - y) * wy
                dist2 = dx*dx + dy*dy

                if dist2 <= threshold * threshold:
                    visited[j] = True
                    queue.append(j)
                    cluster.append(j)

        clusters.append(cluster)

    return clusters

def need_refine_by_distance(centroids, cluster, diag_ratio=1/3):
    """
    檢查 cluster 是否存在某點距離其他點過遠 → 需 refine。
    """
    pts = [centroids[i] for i in cluster]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # cluster bounding box diagonal
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    diag = math.sqrt((w)**2 + (h)**2)

    threshold2 = diag * diag_ratio

    # 檢查每個點與所有其他點的最短距離
    for i, (x1, y1) in enumerate(pts):
        min_dist = float("inf")
        for j, (x2, y2) in enumerate(pts):
            if i == j:
                continue
            dx = (x1 - x2)
            dy = (y1 - y2)
            d = math.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, d)

        # 若與所有人都遠離（大於 diag × ratio）
        if min_dist > threshold2:
            return True

    return False

def check_too_tight(points, tighten_ratio):
    # compute bounding diag
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    diag = math.hypot(dx, dy)
    return diag < tighten_ratio

def hierarchical_cluster(
    centroids,
    base_threshold,
    wx=1.0,
    wy=1.0,
    refine_min_size=4,
    refine_ratio=0.5,
    diag_ratio=1/3,
    tighten_ratio=0.1,     # 新增：點太靠近 → 不 refine
):
    """
    分層式 clustering (遞迴版): 
    1. 第一層用 base_threshold
    2. 若 cluster 點數 ≥ refine_min_size → refine
    3. 若 cluster 內有離群點（最近距離 > diagonal*diag_ratio）→ refine
    4. recurse
    """

    def recurse(cluster_indices, threshold):
        """
        cluster_indices: global indices list
        threshold: current threshold
        """
        pts = [centroids[i] for i in cluster_indices]

        # 第一次 clustering
        sub_clusters = level_cluster_centroids(pts, threshold, wx, wy)

        # sub_clusters 若等於整體 1 cluster，代表已經不能分了
        if len(sub_clusters) == 1:
            sub = sub_clusters[0]
            return [ [cluster_indices[i] for i in sub] ]

        results = []

        for sub in sub_clusters:
            global_sub = [cluster_indices[i] for i in sub]
            pts_sub = [centroids[i] for i in global_sub]

            # === 檢查是否要 refine ===

            # 1. 根據 cluster size
            size_trigger = len(global_sub) >= refine_min_size

            # 2. 根據離群距離
            if diag_ratio >= 0:
                dist_trigger = need_refine_by_distance(centroids, global_sub, diag_ratio)
            else:
                dist_trigger = False

            # 3. 若所有點非常靠近 → 強制不 refine
            if tighten_ratio > 0:
                too_tight = check_too_tight(pts_sub, tighten_ratio)
            else:
                too_tight = False

            # 不 refine → 收下
            if (not size_trigger and not dist_trigger) or too_tight:
                results.append(global_sub)
            else:
                # refine → 進入下一層遞迴
                new_thr = threshold * refine_ratio
                results.extend(recurse(global_sub, new_thr))

        return results

    # 初始叫一次
    return recurse(list(range(len(centroids))), base_threshold)

def level_mark_components_and_clusters_pil(
    input_image: Image.Image,
    wx=1.0,
    wy=1.0, 
    refine_min_size=4,
    refine_ratio=0.5,
    diag_ratio=1/3
):
    """
    input_image: PIL Image
    output: PIL Image
    threshold: 自動由黑色高度決定（高度 125px -> threshold = 50）
    """

    # --- Convert PIL → numpy array ---
    img = np.array(input_image.convert("L"))

    # 黑色前景
    binary = (img != 255).astype(np.uint8)

    # 找黑色塊的垂直範圍（決定 threshold）
    rows = np.any(binary == 1, axis=1)
    ys = np.where(rows)[0]

    if len(ys) == 0:
        black_height = 0
    else:
        black_height = ys.max() - ys.min() + 1
    print(black_height)

    # 你的比例：125px -> 50 threshold
    # threshold = black_height * (50 / 125) = 0.4 * black_height
    threshold = black_height * 0.4

    # --- connected components ---
    structure = np.ones((3,3), dtype=int)
    labeled, num = label(binary, structure=structure)

    # OpenCV RGB image
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    centroids = []
    bboxes = []

    # --- compute centroids & bounding boxes ---
    for i in range(1, num + 1):
        cy, cx = center_of_mass(binary, labels=labeled, index=i)
        cx, cy = int(cx), int(cy)
        centroids.append((cx, cy))

        ys, xs = np.where(labeled == i)
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        bboxes.append((x1, y1, x2, y2))

        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    # --- cluster centroids ---
    clusters = hierarchical_cluster(centroids, threshold, wx, wy, refine_min_size, refine_ratio, diag_ratio=diag_ratio)

    # --- draw cluster boxes ---
    for cluster in clusters:
        xs, ys = [], []
        for idx in cluster:
            x1, y1, x2, y2 = bboxes[idx]
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        X1, Y1 = min(xs), min(ys)
        X2, Y2 = max(xs), max(ys)

        cv2.rectangle(out, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

    # --- Convert numpy → PIL image ---
    output_pil = Image.fromarray(out)

    return output_pil, centroids, clusters, threshold, black_height

