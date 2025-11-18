import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass

def mark_connected_components_centroid(input_path, output_path="out.png"):
    # 讀入灰階
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 黑色視為連通塊（0 = 黑, 255 = 白）
    # 為了方便，把黑色反轉成 1（前景）
    binary = (img != 255).astype(np.uint8)

    # 連通塊標記
    structure = np.ones((3,3), dtype=int)  # 8-connected
    labeled, num = label(binary, structure=structure)

    # 用彩色影像畫點
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    centroids = []

    for i in range(1, num+1):
        cy, cx = center_of_mass(binary, labels=labeled, index=i)
        cx, cy = int(cx), int(cy)
        centroids.append((cx, cy))
        cv2.circle(out, (cx, cy), 3, (0,0,255), -1)  # red dot

    print("Centroids:", centroids)

    cv2.imwrite(output_path, out)
    print("Saved:", output_path)

    return centroids

def cluster_centroids(centroids, threshold, wx=1.0, wy=1.0):
    """
    以質心位置聚類，並允許 x / y 方向不同權重。
    centroids: list of (x, y)
    threshold: clustering distance
    wx, wy: x 與 y 方向權重
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

                # weighted distance
                dx = (cx - x) * wx
                dy = (cy - y) * wy
                dist2 = dx*dx + dy*dy

                if dist2 <= threshold * threshold:
                    visited[j] = True
                    queue.append(j)
                    cluster.append(j)

        clusters.append(cluster)

    return clusters

def mark_components_and_clusters(input_path, output_path="out.png", threshold=20, wx=1.0, wy=1.0):
    # 讀影像
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 黑色前景: 只要不是 255 就算黑
    binary = (img != 255).astype(np.uint8)

    # 找連通塊
    structure = np.ones((3,3), dtype=int)
    labeled, num = label(binary, structure=structure)

    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    centroids = []
    bboxes = []   # (x_min, y_min, x_max, y_max)

    # ----- 取得各連通塊質心與 bounding box -----
    for i in range(1, num + 1):
        cy, cx = center_of_mass(binary, labels=labeled, index=i)
        cx, cy = int(cx), int(cy)
        centroids.append((cx, cy))

        ys, xs = np.where(labeled == i)
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        bboxes.append((x1, y1, x2, y2))

        # 畫紅點（質心）
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    # ----- 聚類質心 -----
    clusters = cluster_centroids(centroids, threshold, wx, wy)

    # ----- 每個 cluster 畫綠色 bounding box -----
    for cluster in clusters:
        xs = []
        ys = []
        for idx in cluster:
            x1, y1, x2, y2 = bboxes[idx]
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        # cluster 的總 box
        X1, Y1 = min(xs), min(ys)
        X2, Y2 = max(xs), max(ys)

        cv2.rectangle(out, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

    cv2.imwrite(output_path, out)
    print("Saved:", output_path)
    return centroids, clusters


if __name__ == "__main__":
    # result = mark_connected_components_centroid("in1.png", "out1.png")
    # result = mark_connected_components_centroid("in2.png", "out2.png")
    # result = mark_connected_components_centroid_cv2("test3.png", "out2.png")
    centroids, clusters = mark_components_and_clusters(
        "in1.png",
        threshold=50, 
        output_path="out1.png", 
        wx=3, 
        wy=0.6
    )
    centroids, clusters = mark_components_and_clusters(
        "in2.png",
        threshold=50, 
        output_path="out2.png", 
        wx=3, 
        wy=0.6
    )
