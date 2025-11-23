import numpy as np
import cv2
import math
from PIL import Image
from scipy.ndimage import label, center_of_mass
from typing import List, Tuple

class Component:
    """
    表示單一連通塊（connected component）。
    - label_id: 原始 labeled index (from scipy.ndimage.label)
    - centroid: (cx, cy) in absolute image coordinates (ints)
    - bbox_xyxy: (x1, y1, x2, y2) absolute coordinates inclusive
    - bbox_yxhw: (y, x, h, w) relative format (方便使用)
    - area: number of pixels
    - mask: cropped boolean mask (h, w) where True indicates foreground within the crop
    - pixels: cropped grayscale pixels (h, w) (原圖灰階值)
    - coords: list/array of (y, x) pixel coords in absolute image coordinates
    """
    def __init__(self,
                 label_id: int,
                 centroid: Tuple[float, float],
                 coords: np.ndarray,
                 full_gray_image: np.ndarray):
        self.label_id = label_id
        # center_of_mass gives (cy, cx)
        cy_f, cx_f = centroid
        self.centroid = (int(round(cx_f)), int(round(cy_f)))  # (cx, cy) as ints

        # coords is Nx2 array of (y, x)
        self.coords = np.asarray(coords, dtype=int)
        if self.coords.size == 0:
            # empty component
            self.area = 0
            self.bbox_xyxy = (0,0,-1,-1)
            self.bbox_yxhw = (0,0,0,0)
            self.mask = np.zeros((0,0), dtype=bool)
            self.pixels = np.zeros((0,0), dtype=full_gray_image.dtype)
            return

        ys = self.coords[:,0]
        xs = self.coords[:,1]
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        # inclusive bbox
        self.bbox_xyxy = (x1, y1, x2, y2)
        h = y2 - y1 + 1
        w = x1 and (x2 - x1 + 1) or (x2 - x1 + 1)  # robust but simpler below
        # simpler:
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        self.bbox_yxhw = (y1, x1, h, w)
        self.area = int(len(self.coords))

        # cropped grayscale pixels and mask
        cropped = full_gray_image[y1:y2+1, x1:x2+1]
        mask = np.zeros_like(cropped, dtype=bool)
        # coords relative to crop:
        rel_y = ys - y1
        rel_x = xs - x1
        mask[rel_y, rel_x] = True

        self.mask = mask
        self.pixels = cropped
        # store centroid as floats too if needed
        self.centroid_float = (cx_f, cy_f)

    def get_centroid(self) -> Tuple[int,int]:
        return self.centroid

    def get_bbox_yxhw(self) -> Tuple[int,int,int,int]:
        return self.bbox_yxhw

    def get_bbox_xyxy(self) -> Tuple[int,int,int,int]:
        return self.bbox_xyxy

class ClusterGroup:
    """
    代表一個 cluster，由多個 Component 組成。
    會計算：
    - components: list[Component]
    - bbox_yxhw: aggregate bbox (y,x,h,w)
    - bbox_xyxy: (x1,y1,x2,y2)
    - area: sum of areas
    - centroid: weighted centroid (by area), integer (cx,cy)
    - mask_on_full: optional combined mask in full image coordinates (bool array) — 建立時可選擇開啟
    """
    def __init__(self, components: List[Component], build_full_mask: bool=False, full_shape: Tuple[int,int] | None=None):
        self.components = list(components)
        if len(self.components) == 0:
            self.bbox_xyxy = (0,0,-1,-1)
            self.bbox_yxhw = (0,0,0,0)
            self.area = 0
            self.centroid = (0,0)
            self.mask_on_full = None
            return

        # aggregate bbox
        x1s = [c.get_bbox_xyxy()[0] for c in self.components]
        y1s = [c.get_bbox_xyxy()[1] for c in self.components]
        x2s = [c.get_bbox_xyxy()[2] for c in self.components]
        y2s = [c.get_bbox_xyxy()[3] for c in self.components]

        X1, Y1 = min(x1s), min(y1s)
        X2, Y2 = max(x2s), max(y2s)
        self.bbox_xyxy = (X1, Y1, X2, Y2)
        h = Y2 - Y1 + 1
        w = X2 - X1 + 1
        self.bbox_yxhw = (Y1, X1, h, w)

        # area & weighted centroid
        areas = np.array([c.area for c in self.components], dtype=float)
        self.area = int(areas.sum())
        if self.area > 0:
            # compute weighted centroid: use float centroids (cx,cy)
            cxs = np.array([c.centroid_float[0] for c in self.components], dtype=float)
            cys = np.array([c.centroid_float[1] for c in self.components], dtype=float)
            cx = float((cxs * areas).sum() / areas.sum())
            cy = float((cys * areas).sum() / areas.sum())
            self.centroid = (int(round(cx)), int(round(cy)))
            self.centroid_float = (cx, cy)
        else:
            self.centroid = (0,0)
            self.centroid_float = (0.0, 0.0)

        # optionally build full-image mask
        self.mask_on_full = None
        if build_full_mask:
            if full_shape is None:
                raise ValueError("full_shape must be provided to build full mask")
            hfull, wfull = full_shape
            mask = np.zeros((hfull, wfull), dtype=bool)
            for c in self.components:
                y1, x1, h, w = c.get_bbox_yxhw()
                mask[y1:y1+h, x1:x1+w] |= c.mask
            self.mask_on_full = mask

    def get_bbox_yxhw(self) -> Tuple[int,int,int,int]:
        return self.bbox_yxhw

    def get_centroid(self) -> Tuple[int,int]:
        return self.centroid

# ----- clustering utilities (operate on centroids lists) -----
def level_cluster_centroids(centroids: List[Tuple[float,float]], threshold: float, wx=1.0, wy=1.0) -> List[List[int]]:
    """
    基本的 cluster (BFS) 基於加權歐式距離。輸入 centroids 為 [(cx,cy), ...]。
    回傳：list of clusters，cluster 是 index list。
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

def check_too_tight(points: List[Tuple[float,float]], tighten_ratio: float):
    """
    判斷 cluster 裡的點是否非常靠近：用 bbox 對角線與 tighten_ratio 做比較。
    points: list of (x,y) or (cx,cy)
    若 diag < tighten_ratio -> too tight
    """
    if len(points) == 0:
        return True
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    diag = math.hypot(dx, dy)
    return diag < tighten_ratio

def hierarchical_cluster(
    components: List[Component],
    base_threshold: float,
    wx=1.0,
    wy=1.0,
    refine_min_size=4,
    refine_ratio=0.5,
    tighten_ratio=0.1,
    build_cluster_masks: bool=False,
    image_shape: Tuple[int,int] | None=None
) -> List[ClusterGroup]:
    """
    分層式 clustering，輸入 components 列表（每個有 centroid）。
    參數語意與原來程式相同。
    回傳：ClusterGroup list（每個包含其 components 與聚合資訊）。
    若 build_cluster_masks=True，則會為每個 cluster 建立 full-image mask（需要 image_shape）。
    """

    # 提取 centroids 簡化運算
    centroids = [c.centroid_float for c in components]

    def recurse(index_list: List[int], threshold: float) -> List[List[int]]:
        """
        index_list: list of indices into components
        threshold: current threshold
        返回分好的小 cluster 的 index lists（局部）
        """
        if len(index_list) == 0:
            return []
        pts = [centroids[i] for i in index_list]
        sub_clusters = level_cluster_centroids(pts, threshold, wx, wy)

        # 如果只有一個 sub_cluster 且其包含全部點，回傳該 cluster
        if len(sub_clusters) == 1:
            sub = sub_clusters[0]
            return [[index_list[i] for i in sub]]

        results = []
        for sub in sub_clusters:
            global_sub = [index_list[i] for i in sub]
            pts_sub = [centroids[i] for i in global_sub]

            size_trigger = len(global_sub) >= refine_min_size

            if tighten_ratio > 0:
                too_tight = check_too_tight(pts_sub, tighten_ratio)
            else:
                too_tight = False

            if (not size_trigger) or too_tight:
                results.append(global_sub)
            else:
                new_thr = threshold * refine_ratio
                results.extend(recurse(global_sub, new_thr))

        return results

    top_index_list = list(range(len(components)))
    partitioned_index_lists = recurse(top_index_list, base_threshold)

    # build ClusterGroup objects
    clusters = []
    for idx_list in partitioned_index_lists:
        comps = [components[i] for i in idx_list]
        cluster = ClusterGroup(comps, build_full_mask=build_cluster_masks, full_shape=image_shape)
        clusters.append(cluster)

    return clusters

# ----- 主流程：基於 Component 物件 -----
def extract_components_from_pil(
    input_image: Image.Image,
    binary_threshold: int = 127,
) -> Tuple[List[Component], int, np.ndarray]:
    """
    從 PIL Image 中找出 connected components 並回傳 Component 物件列表。
    - binary_threshold: 灰階<=threshold視為前景
    - 返回 (components, black_height, gray_image)
    black_height: 前景在 y 軸上的高度（用來決定 threshold）
    """
    gray = np.array(input_image.convert("L"))
    binary = (gray <= binary_threshold).astype(np.uint8)

    # 找黑色塊的垂直範圍（決定 black_height）
    rows = np.any(binary == 1, axis=1)
    ys = np.where(rows)[0]
    if len(ys) == 0:
        black_height = 0
    else:
        black_height = int(ys.max() - ys.min() + 1)

    # connected components
    structure = np.ones((3,3), dtype=int)
    labeled, num = label(binary, structure=structure) # type: ignore

    components = []
    for i in range(1, num + 1):
        # coordinates
        ys_i, xs_i = np.where(labeled == i)
        coords = np.vstack([ys_i, xs_i]).T  # Nx2 (y,x)
        if coords.size == 0:
            continue
        # centroid
        cy, cx = center_of_mass(binary, labels=labeled, index=i)
        comp = Component(label_id=i, centroid=(cy, cx), coords=coords, full_gray_image=gray) # type: ignore
        components.append(comp)

    return components, black_height, gray

def level_mark_components_and_clusters_pil(
    input_image: Image.Image,
    wx=1.0,
    wy=1.0,
    refine_min_size=4,
    refine_ratio=0.5,
    tighten_ratio=0.1,
    binary_threshold: int = 127,
):
    """
    與原先功能等價但以 Component/ClusterGroup 為主體。
    回傳：
    - output_pil: 繪製結果的 PIL image (BGR->RGB 輸出)
    - components: list[Component]
    - clusters: list[ClusterGroup]
    - boxes: list of (y,x,h,w) for each cluster
    - black_height: 計算到的黑色區域高度（用來決定 base_threshold）
    """
    # extract components
    components, black_height, gray = extract_components_from_pil(input_image, binary_threshold=binary_threshold)

    # threshold 決定（維持原始邏輯）
    base_threshold = black_height * 0.4

    # clustering
    clusters = hierarchical_cluster(
        components,
        base_threshold=base_threshold,
        wx=wx,
        wy=wy,
        refine_min_size=refine_min_size,
        refine_ratio=refine_ratio,
        tighten_ratio=tighten_ratio,
        build_cluster_masks=False,
        image_shape=gray.shape
    )

    # draw output (BGR)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # draw centroids and individual bboxes (optional)
    for comp in components:
        cx, cy = comp.get_centroid()
        cv2.circle(out, (cx, cy), 2, (0, 0, 255), -1)
        # draw small bbox (component)
        x1, y1, x2, y2 = comp.get_bbox_xyxy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 1)

    boxes: List[Tuple[int, int, int, int]] = []
    # draw cluster boxes
    for cluster in clusters:
        Y1, X1, h, w = cluster.get_bbox_yxhw()
        Y2 = Y1 + h - 1
        X2 = X1 + w - 1
        boxes.append((Y1, X1, h, w))
        cv2.rectangle(out, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
        # draw cluster centroid
        cx, cy = cluster.get_centroid()
        cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1)

    output_pil = Image.fromarray(out)

    return output_pil, components, clusters, boxes, black_height

# ------------------------
# 使用範例（示範）
# ------------------------
if __name__ == "__main__":
    # 範例：載入一張圖，並執行
    img = Image.open("test.png")  # 改成你的檔案
    out_img, components, clusters, boxes, black_h = level_mark_components_and_clusters_pil(
        img, wx=1.0, wy=1.0, refine_min_size=4, refine_ratio=0.5, tighten_ratio=0.1
    )
    print("Components:", len(components))
    print("Clusters:", len(clusters))
    print("Boxes (y,x,h,w):", boxes)
    out_img.save("debug_out.png")
