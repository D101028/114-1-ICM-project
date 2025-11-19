import numpy as np
import cv2
from PIL import Image
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
    以質心位置聚類，允許 x / y 方向不同權重。
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

                dx = (cx - x) * wx
                dy = (cy - y) * wy
                dist2 = dx*dx + dy*dy

                if dist2 <= threshold * threshold:
                    visited[j] = True
                    queue.append(j)
                    cluster.append(j)

        clusters.append(cluster)

    return clusters

def mark_components_and_clusters_pil(
    input_image: Image.Image,
    wx=1.0,
    wy=1.0
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
    clusters = cluster_centroids(centroids, threshold, wx, wy)

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


if __name__ == "__main__":
    # result = mark_connected_components_centroid("in1.png", "out1.png")
    # result = mark_connected_components_centroid("in2.png", "out2.png")
    # result = mark_connected_components_centroid_cv2("test3.png", "out2.png")
    img1 = Image.open("in1.png")
    out_img = mark_components_and_clusters_pil(img1, wx=3, wy=0.6)[0]
    out_img.save("out1.png")
    img1 = Image.open("in2.png")
    out_img = mark_components_and_clusters_pil(img1, wx=2.5, wy=1.2)[0]
    out_img.save("out2.png")
    # centroids, clusters = mark_components_and_clusters(
    #     "in1.png",
    #     threshold=50, 
    #     output_path="out1.png", 
    #     wx=3, 
    #     wy=0.6
    # )
    # centroids, clusters = mark_components_and_clusters(
    #     "in2.png",
    #     threshold=50, 
    #     output_path="out2.png", 
    #     wx=3, 
    #     wy=0.6
    # )
