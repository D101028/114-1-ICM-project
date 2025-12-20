from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import label, center_of_mass

class Component:
    """
    表示單一連通塊（connected component）。
    - centroid: (cx, cy) in absolute image coordinates (ints)
    - bbox_xyxy: (x1, y1, x2, y2) absolute coordinates inclusive
    - bbox_yxhw: (y, x, h, w) relative format (方便使用)
    - area: number of pixels
    - mask: cropped boolean mask (h, w) where True indicates foreground within the crop
    - pixels: cropped grayscale pixels (h, w) (原圖灰階值)
    - coords: list/array of (y, x) pixel coords in absolute image coordinates
    """
    def __init__(self,
                 centroid: Tuple[float, float],
                 coords: np.ndarray,
                 full_gray_image: np.ndarray):
        # center_of_mass gives (cy, cx)
        cy_f, cx_f = centroid
        self.centroid = (int(round(cx_f)), int(round(cy_f)))  # (cx, cy) as ints

        self.full_gray_image = full_gray_image

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

        self.soft_contour_points: np.ndarray | None = None

    def get_centroid(self) -> Tuple[int,int]:
        return self.centroid

    def get_bbox_yxhw(self) -> Tuple[int,int,int,int]:
        return self.bbox_yxhw

    def get_bbox_xyxy(self) -> Tuple[int,int,int,int]:
        return self.bbox_xyxy

    def copy(self) -> "Component":
        return Component(self.centroid, self.coords, self.full_gray_image)

    def move(self, dy: int, dx: int):
        self.centroid = (self.centroid[0] + dx, self.centroid[1] + dy)
        self.bbox_xyxy = (self.bbox_xyxy[0] + dx, self.bbox_xyxy[1] + dy, self.bbox_xyxy[2] + dx, self.bbox_xyxy[3] + dy)
        self.bbox_yxhw = (self.bbox_yxhw[0] + dy, self.bbox_yxhw[1] + dx, self.bbox_yxhw[2], self.bbox_yxhw[3])

    def get_soft_contour_points(self, thresh: int = 127) -> np.ndarray:
        """
        灰階深淺變化的輪廓。
        若 pixel 與任一 4-neighbor 的灰階差 > thresh，就視為輪廓。
        
        邊界以外視為 255，允許座標為負。
        
        回傳：絕對座標 (y, x) 的 ndarray (K, 2)
        """
        if self.soft_contour_points is not None:
            return self.soft_contour_points

        mask = self.mask
        pixels = self.pixels

        if mask.size == 0:
            return np.zeros((0, 2), dtype=int)

        h, w = mask.shape
        ys, xs = np.where(mask)

        contour = []
        y0, x0, _, _ = self.bbox_yxhw

        for y, x in zip(ys, xs):
            p = pixels[y, x]

            neighbors = []

            # 上
            if y > 0:
                neighbors.append(pixels[y-1, x])
            else:
                neighbors.append(255)  # 上界外
            # 下
            if y < h - 1:
                neighbors.append(pixels[y+1, x])
            else:
                neighbors.append(255)  # 下界外
            # 左
            if x > 0:
                neighbors.append(pixels[y, x-1])
            else:
                neighbors.append(255)  # 左界外
            # 右
            if x < w - 1:
                neighbors.append(pixels[y, x+1])
            else:
                neighbors.append(255)  # 右界外

            # 若與任一鄰居亮度差 > threshold，即視為輪廓
            if any(abs(int(p) - int(nb)) > thresh for nb in neighbors):
                contour.append((y + y0, x + x0))

        self.soft_contour_points = np.array(contour, dtype=int)

        return self.soft_contour_points

    def pixels_below_threshold(self, thres: int) -> np.ndarray:
        """
        回傳所有灰階值大於 thres 的像素在絕對座標下的 ndarray (N, 2)。
        座標格式為 (y, x)。
        """
        # mask 大於閾值的相對座標
        rel_mask = self.pixels < thres
        if not rel_mask.any():
            return np.zeros((0,2), dtype=int)
        
        rel_coords = np.argwhere(rel_mask)  # relative coords within crop
        # 轉換成絕對座標
        y0, x0, _, _ = self.bbox_yxhw
        abs_coords = rel_coords + np.array([y0, x0])
        return abs_coords

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
    def __init__(
        self, components: List[Component], 
        build_full_mask: bool=False, 
        full_shape: Tuple[int,int] | None=None, 
        topleft: Tuple[int,int]=(0, 0)
    ):
        self.components = [c.copy() for c in components]
        if len(self.components) == 0:
            self.bbox_hw = (0,0)
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
        h = Y2 - Y1 + 1
        w = X2 - X1 + 1
        self.bbox_hw = (h, w)
        self.topleft = (topleft[0] + Y1, topleft[1] + X1)
        
        # correct the location of components
        for c in self.components:
            c.move(-Y1, -X1)

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
        
        self.l_img: Image.Image | None = None
        self.soft_contour_arr: np.ndarray | None = None
        self.pixels_below_threshold_arr: np.ndarray | None = None

    def to_L(self) -> Image.Image:
        if self.l_img is not None:
            return self.l_img

        h, w = self.get_bbox_hw()
        # create a plain grayscale canvas (no alpha)
        canvas = Image.new("L", (w, h), 255)

        for comp in self.components:
            cy1, cx1, ch, cw = comp.get_bbox_yxhw()

            mask = comp.mask
            pix = comp.pixels  # grayscale array (h,w)

            # component grayscale image
            comp_img = Image.fromarray(pix.astype(np.uint8), mode="L")
            # mask to control where to paste (still single-channel, no alpha in result)
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

            canvas.paste(comp_img, (cx1, cy1), mask_img)
        
        self.l_img = canvas

        return canvas

    def get_bbox_hw(self) -> Tuple[int,int]:
        return self.bbox_hw

    def get_centroid(self) -> Tuple[int,int]:
        """Return centroid in (x, y)"""
        return self.centroid

    def get_soft_contour_arr(self) -> np.ndarray:
        if self.soft_contour_arr is not None:
            return self.soft_contour_arr
        arr = np.empty((0, 2), dtype=int)
        for comp in self.components:
            pts = comp.get_soft_contour_points()   # shape (K,2)
            if pts.size == 0:
                continue
            arr = np.vstack([arr, pts])
        self.soft_contour_arr = arr
        return arr
    
    def get_pixels_below_threshold(self, threshold=127) -> np.ndarray:
        if self.pixels_below_threshold_arr is not None:
            return self.pixels_below_threshold_arr
        
        arr = np.empty((0, 2), dtype=int)
        for comp in self.components:
            pts = comp.pixels_below_threshold(127)
            if pts.size == 0:
                continue
            arr = np.vstack([arr, pts])
        self.pixels_below_threshold_arr = arr
        
        return arr

    def resize(self, size: Tuple[int, int]) -> 'ClusterGroup':
        """
        縮放整個 ClusterGroup 並回傳一個新的 ClusterGroup 實例。\\
        縮放後的 ClusterGroup 不保留 topleft 資訊。
        
        Args:
            size (Tuple[int, int]): 目標尺寸 (width, height)
        Returns:
            ClusterGroup: 縮放後的新 ClusterGroup 物件
        """
        target_w, target_h = size
        orig_h, orig_w = self.get_bbox_hw()
        
        if orig_w == 0 or orig_h == 0:
            return ClusterGroup([])

        # 1. 計算縮放倍率
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        new_components = []

        for comp in self.components:
            # 計算該組件在縮放後的新尺寸
            c_y1, c_x1, c_h, c_w = comp.bbox_yxhw
            
            # 縮放後的尺寸與位置
            new_cw = max(1, int(round(c_w * scale_x)))
            new_ch = max(1, int(round(c_h * scale_y)))
            new_cx1 = int(round(c_x1 * scale_x))
            new_cy1 = int(round(c_y1 * scale_y))

            # 2. 縮放 Pixels 與 Mask
            # 使用 PIL 進行影像處理
            img_pix = Image.fromarray(comp.pixels)
            img_mask = Image.fromarray(comp.mask.astype(np.uint8) * 255)
            
            res_pix = np.array(img_pix.resize((new_cw, new_ch), Image.Resampling.LANCZOS))
            res_mask = np.array(img_mask.resize((new_cw, new_ch), Image.Resampling.NEAREST)) > 127

            # 3. 重建座標 (coords)
            # 因為 Component.__init__ 依賴 coords 來計算 bbox 和 mask
            # 我們從 res_mask 中提取局部座標，再轉換為「偽全圖」座標
            local_ys, local_xs = np.where(res_mask)
            # 這裡的座標是相對於新 Cluster 左上角 (0,0) 的
            new_coords = np.column_stack((local_ys + new_cy1, local_xs + new_cx1))

            # 4. 建立新的 Component
            # 注意：這裡我們提供一個「局部背景圖」給 Component init 
            # 為了符合你 Component 的設計，我們建立一個剛好涵蓋該 component 的畫布
            fake_full_gray = np.zeros((new_cy1 + new_ch, new_cx1 + new_cw), dtype=res_pix.dtype)
            fake_full_gray[new_cy1:new_cy1+new_ch, new_cx1:new_cx1+new_cw] = res_pix
            
            # 計算新的質心 (簡單縮放)
            new_centroid = (comp.centroid_float[0] * scale_x, comp.centroid_float[1] * scale_y)
            
            new_comp = Component(
                centroid=new_centroid,
                coords=new_coords,
                full_gray_image=fake_full_gray
            )
            new_components.append(new_comp)

        # 回傳新的 ClusterGroup
        return ClusterGroup(new_components)

def level_cluster_centroids(centroids: List[Tuple[float,float]], threshold: float, wx=1.0, wy=1.0) -> List[List[int]]:
    """
    對 centroids 做 cluster (BFS) based on 加權歐式距離 \\sqrt{wx\\*dx + wy\\*dy}。
    - 輸入 centroids 為 [(cx,cy), ...]。
    - 回傳：list of clusters，cluster 是 index list。
    """
    n = len(centroids)
    visited = [False] * n
    clusters = []

    sqr_thr = threshold * threshold

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
                if dist2 <= sqr_thr:
                    visited[j] = True
                    queue.append(j)
                    cluster.append(j)

        clusters.append(cluster)

    return clusters

def hierarchical_cluster(
    components: List[Component],
    base_threshold: float,
    wx=1.0,
    wy=1.0,
    refine_min_size=4,
    refine_ratio=0.5,
    max_depth=12, 
    build_cluster_masks: bool=False,
    image_shape: Tuple[int,int] | None=None, 
    topleft: Tuple[int, int]=(0, 0)
) -> List[ClusterGroup]:
    """
    分層式 clustering。對 components 的 centroids 組成做 cluster。
    若 each cluster 有超過 refine_min_size 個 centroids，則 refine ratio 後遞迴，直至低於 refine_min_size 或者超過遞迴深度。
    
    - 輸入： components 列表（每個有 centroid）。
    - 回傳：ClusterGroup list（每個包含其 components 與聚合資訊）。
    
    若 build_cluster_masks=True，則會為每個 cluster 建立 full-image mask（需要 image_shape）。
    """

    # 提取 centroids 簡化運算
    centroids = [c.centroid_float for c in components]

    def recurse(index_list: List[int], threshold: float, depth: int = 0) -> List[List[int]]:
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

            if len(global_sub) < refine_min_size or depth >= max_depth:
                results.append(global_sub)
            else:
                new_thr = threshold * refine_ratio
                results.extend(recurse(global_sub, new_thr, depth+1))

        return results

    top_index_list = list(range(len(components)))
    partitioned_index_lists = recurse(top_index_list, base_threshold)

    # build ClusterGroup objects
    clusters: List[ClusterGroup] = []
    for idx_list in partitioned_index_lists:
        comps = [components[i] for i in idx_list]
        cluster = ClusterGroup(comps, build_full_mask=build_cluster_masks, full_shape=image_shape, topleft=topleft)
        clusters.append(cluster)

    return clusters

def extract_components_from_pil(
    input_image: Image.Image,
    binary_threshold: int = 127
) -> Tuple[List[Component], Tuple[int, int, int, int], np.ndarray]:
    """
    從 PIL Image 中找出 connected components 並回傳 Component 物件列表。
    - binary_threshold: 灰階<=threshold視為前景
    - 返回 (components, bbox_yxhw, gray_image)
    """
    gray = np.array(input_image.convert("L"))
    binary = (gray <= binary_threshold).astype(np.uint8)

    # 找黑色塊的完整 bbox (y, x, h, w)
    rows = np.any(binary == 1, axis=1)
    cols = np.any(binary == 1, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    if len(ys) == 0 or len(xs) == 0:
        bbox_yxhw = (0, 0, 0, 0)
    else:
        y1 = int(ys.min())
        y2 = int(ys.max())
        x1 = int(xs.min())
        x2 = int(xs.max())
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        bbox_yxhw = (y1, x1, h, w)

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
        comp = Component(centroid=(cy, cx), coords=coords, full_gray_image=gray) # type: ignore
        components.append(comp)

    return components, bbox_yxhw, gray
