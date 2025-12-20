import numpy as np
import cv2
import os
import sys
from PIL import Image, ImageEnhance, ImageDraw
from prettytable import PrettyTable
from typing import List, Tuple, Dict
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from macrotopng import latex_symbol_to_png
from mc import level_mark_components_and_clusters_pil as lmccp2
from mc import ClusterGroup, extract_components_from_pil, hierarchical_cluster
from pixelwise import dist_compare

def test0():
    # texs = [
    #     r"e^{i\pi^{jk^{ij}}}", 
    #     r"\text{Apple, I have icecream } \sum_{i=0}^{2^{10}}e^{-i}", 
    #     r"j \cdot \sum^{i}", 
    #     r"\Theta \varTheta \Xi \varXi \| \Vert \lim_{x \to \infty} \sqrt[n]{abc}\circledast \circledcirc \div \doublebarwedge \doteqdot \fallingdotseq \risingdotseq \Vvdash ", 
    #     r"\text{Apple, I have icecream } ", 
    #     r"i\quad j\quad := \doteqdot \div \doteq "
    # ]
    texs = [
        r"jiii", 
        r"jiii \cdot \sum_{ij}^{jj}"
        r"j \cdot \sum^{i}", 
        r"j\sum", 
        r"jjiii \cdot \sum_{ij}^{jj^{ij}}", 
    ]

    for idx, tex in enumerate(texs):
        latex_symbol_to_png(
            tex, 
            dpi=600, 
            out_path=f"./data/in{idx}.png", 
            background=(255,255,255)
        )

def test3():
    import random
    def create_img(macro: str):
        name = macro.replace("\\", "_bs_").replace("|", "_vert_") + "_" + str(random.randint(0, 100)) + ".png"
        if name[0] == "_":
            name = name[1:]
        elif name[0] == ":":
            name = "colon.png"
        print(name)
        fp = f"templates/{name}"
        latex_symbol_to_png(
            macro, 
            out_path = fp, 
            background = (255,255,255)
        )
        img = ImageEnhance.Contrast(Image.open(fp).convert("L")).enhance(4.0)
        from pixelwise import _crop_img_obj
        _crop_img_obj(img, (255,255,255)).convert("L").save(fp)
    create_img("|")
    return 
    samples = r"""
0123456789
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
\text{a} \text{b} \text{c} \text{d} \text{e} \text{f} \text{g} \text{h} \text{i} \text{j} \text{k} \text{l} \text{m} \text{n} \text{o} \text{p} \text{q} \text{r} \text{s} \text{t} \text{u} \text{v} \text{w} \text{x} \text{y} \text{z}
\text{A} \text{B} \text{C} \text{D} \text{E} \text{F} \text{G} \text{H} \text{I} \text{J} \text{K} \text{L} \text{M} \text{N} \text{O} \text{P} \text{Q} \text{R} \text{S} \text{T} \text{U} \text{V} \text{W} \text{X} \text{Y} \text{Z}
\cdot \sum \alpha \beta + - \tims * / \leq \geq | ( ) = [ ] , \int
""".strip().replace("\n", " ")
    for macro in samples.split():
        try:
            if macro[0] != "\\" and len(macro) != 1:
                for m in macro:
                    create_img(m)
            else:
                create_img(macro)
        except Exception as e:
            print(e)

AnsType = str | None
SimType = float
AdaptiveReturnType = List[Tuple[ClusterGroup, AnsType, SimType, Tuple[int, int]]]
SauceType = Dict[AnsType, Tuple[Image.Image, float, ClusterGroup]]

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

def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    # A, B shape: (N, 2), (M, 2)
    if A.size == 0 or B.size == 0:
        return np.inf # 回傳最大值

    # 使用 cdist 計算成對距離矩陣 (N, M)
    dists = cdist(A, B, metric='euclidean')
    
    # Directed distances
    h_A_B = np.max(np.min(dists, axis=1))
    h_B_A = np.max(np.min(dists, axis=0))
    
    return max(h_A_B, h_B_A)

def hausdorff_ratio(base_cluster: ClusterGroup, cluster: ClusterGroup, topleft: Tuple[int, int]) -> float:
    """
    計算兩個 ClusterGroup 的 hausdorff 相似度。
    
    :param base_cluster: templates cluster
    :type base_cluster: ClusterGroup
    :param cluster: 欲比較之 cluster，先縮放到與 base_cluster 相同 size 再比較
    :type cluster: ClusterGroup
    :param topleft: 校準 cluster components 用
    :type topleft: Tuple[int, int]
    :return: [0, 1] 區間的值
    :rtype: float
    """
    # resize 
    _, _, h, w = base_cluster.get_bbox_yxhw()
    cluster = cluster.resize((w, h))

    arr1 = np.empty((0, 2), dtype=int)
    arr2 = np.empty((0, 2), dtype=int)

    for comp in base_cluster.components:
        pts = comp.get_soft_contour_points()   # shape (K,2)
        if pts.size == 0:
            continue
        arr1 = np.vstack([arr1, pts])

    for comp in cluster.components:
        pts = comp.get_soft_contour_points()
        pts[:, 0] = pts[:, 0] - topleft[0]
        pts[:, 1] = pts[:, 1] - topleft[1]
        if pts.size == 0:
            continue
        arr2 = np.vstack([arr2, pts])

    if arr1.size == 0 or arr2.size == 0:
        return 0.0 # 若為空，返回 0 相似

    dist = hausdorff_distance(arr1, arr2)
    
    # 方案 A：使用兩者聯集的對角線進行標準化 (最推薦，確保 ratio <= 1)
    all_pts = np.vstack([arr1, arr2])
    min_coords = all_pts.min(axis=0)
    max_coords = all_pts.max(axis=0)
    diff = max_coords - min_coords
    diagonal = np.sqrt(np.sum(diff**2))
    
    # 避免除以零
    if diagonal == 0:
        return 0.0 if dist == 0 else 1.0
        
    ratio = dist / diagonal
    return 1 - ratio

def l2_ratio(cluster1: ClusterGroup, cluster2: ClusterGroup, topleft):
    img1 = cluster1.to_L()
    img2 = cluster2.to_L()

    arr1 = np.asarray(img1, dtype=np.float32).ravel()
    arr2 = np.asarray(img2, dtype=np.float32).ravel()

    l2_distance = np.linalg.norm(arr1 - arr2, ord=2)
    max_distance = np.sqrt(len(arr1)) * 255  # Maximum possible distance

    similarity = 1 - (l2_distance / max_distance)
    return float(np.clip(similarity, 0, 1)) # ensure the value is in [0, 1]

def adaptive_cluster(
    img: Image.Image, sauce: SauceType, accept_sim = 0.7, accept_h_sim = 0.9, max_depth = 16
) -> AdaptiveReturnType:

    def recurse(img: Image.Image, depth = 0, fix_ratio_x = 1.0, fix_ratio_y = 1.0, topleft = (0, 0)) -> AdaptiveReturnType:
        # 1. 初始化參數與分群
        wx, wy = 2 + 0.1 * fix_ratio_x, 0.1 + 0.1 * fix_ratio_y
        components, bbox_yxhw, gray = extract_components_from_pil(img)
        
        clusters = hierarchical_cluster(
            components,
            base_threshold=bbox_yxhw[2] * 0.4,
            wx=wx, wy=wy,
            refine_min_size=4, refine_ratio=0.5,
            build_cluster_masks=False,
            image_shape=gray.shape
        )

        # 2. 特殊情況：如果分不開且還有深度，增加權重重試
        if len(clusters) == 1 and len(clusters[0].components) > 1 and depth < max_depth:
            return recurse(img, depth + 1, fix_ratio_x + 0.5, fix_ratio_y + 2, topleft)
        
        rec_out: AdaptiveReturnType = []
        for cluster in clusters:
            y, x, h, w = cluster.get_bbox_yxhw()
            next_topleft = (topleft[0] + y, topleft[1] + x)
            r = h / w
            tgt = cluster.to_L()

            # 3. 遞迴深挖：如果組件過多，強制進入下一層
            if len(cluster.components) > 4 and depth < max_depth:
                rec_out.extend(recurse(tgt, depth + 1, topleft=next_topleft))
                continue

            # 4. 統一比對邏輯
            best_f, max_sim, h_sim = None, 0.0, 0.0
            
            for f, (src, yx_ratio, base_cluster) in sauce.items():
                # 比率篩選邏輯
                is_compatible = (
                    (r < 0.1 and yx_ratio <= 0.125) or
                    (r > 10 and yx_ratio >= 8) or
                    (0.1 <= r <= 10 and 0.1 <= yx_ratio <= 10)
                )
                if not is_compatible: continue

                l2_sim = dist_compare(src, tgt, (255, 255, 255))
                
                if l2_sim > max_sim:
                    max_sim = l2_sim
                    h_sim = hausdorff_ratio(base_cluster, cluster, topleft)
                    best_f = f

            # 5. 判定與輸出
            # 如果相似度不足且還有組件，嘗試進一步拆解
            if (max_sim < accept_sim or h_sim < accept_h_sim) and depth < max_depth and len(cluster.components) > 1:
                if str(h_sim) == "0.10932623560103483":
                    print(h_sim, best_f)
                    if best_f is not None and "leq" in best_f:
                        cluster.to_L().save("out0.png")
                        sauce[best_f][0].save("out1.png")
                rec_out.extend(recurse(tgt, depth + 1, topleft=next_topleft))
            else:
                if max_sim < accept_sim and best_f is not None:
                    print(f"Doubted: {best_f} (Sim: {max_sim:.3f})")
                rec_out.append((cluster, best_f, max_sim, topleft))
                
        return rec_out
    
    return recurse(img)

def test4():
    sauce: SauceType = {}
    for root, dirs, files in os.walk("./templates"):
        for f in files:
            path = f"{root}/{f}"
            src = Image.open(path)
            components, bbox_yxhw, gray_image = extract_components_from_pil(src)
            cluster = ClusterGroup(components)
            sauce[f] = (src, src.size[1] / src.size[0], cluster)
    
    src_img = Image.open(sys.argv[1])
    src_img = src_img.convert("L")
    enhancer = ImageEnhance.Contrast(src_img)
    src_img = enhancer.enhance(2.0)

    import time 
    start = time.time()
    out = adaptive_cluster(src_img, sauce)
    print(time.time() - start)

    src_img = src_img.convert("RGB")
    myTable = PrettyTable(["Position (y, x, h, w)", "Centroid (x, y)", "Answer", "Similarity"])
    tableRows = []
    for cluster, ans, sim, topleft in out:
        dy, dx = topleft
        y, x, h, w = cluster.get_bbox_yxhw()
        cx, cy = cluster.get_centroid()
        tableRows.append([
            (y+dy, x+dx, h, w), 
            (cx+dx, cy+dy), 
            ans, sim
        ])
        draw = ImageDraw.Draw(src_img, "RGB")
        draw.rectangle((x+dx-1, y+dy-1, x+dx+w+1, y+dy+h+1), None, (0,0,128) if sim >= 0.7 else (128,0,0))
    tableRows.sort(
        key = lambda row: row[1][0]
    )
    for row in tableRows:
        myTable.add_row(row)
    src_img.save("test.png")
    print(myTable)

def test1():
    img0 = Image.open("a1_resized.png")
    img1 = Image.open("a1.png")
    img2 = Image.open("a2.png")
    img3 = Image.open("a3.png")
    comp0, _, _ = extract_components_from_pil(img0)
    comp1, _, _ = extract_components_from_pil(img1)
    comp2, _, _ = extract_components_from_pil(img2)
    comp3, _, _ = extract_components_from_pil(img3)

    cluster0 = ClusterGroup(comp0)
    _, _, h, w = cluster0.get_bbox_yxhw()
    cluster1 = ClusterGroup(comp1).resize((w, h))
    cluster2 = ClusterGroup(comp2).resize((w, h))
    cluster3 = ClusterGroup(comp3).resize((w, h))
    
    ratio0 = chamfer_ratio(cluster1, cluster0, (0, 0))
    ratio1 = chamfer_ratio(cluster1, cluster1, (0, 0))
    ratio2 = chamfer_ratio(cluster1, cluster2, (0, 0))
    ratio3 = chamfer_ratio(cluster1, cluster3, (0, 0))
    
    print(ratio0, ratio1, ratio2, ratio3)

def test2():
    img1 = Image.open("out0.png")
    img2 = Image.open("out1.png")
    temp = Image.open("templates/bs_leq_79.png")
    
    comp, _, _ = extract_components_from_pil(temp)
    comp1, _, _ = extract_components_from_pil(img1)
    comp2, _, _ = extract_components_from_pil(img2)

    cluster = ClusterGroup(comp)
    _, _, h, w = cluster.get_bbox_yxhw()
    cluster1 = ClusterGroup(comp1).resize((w, h))
    cluster2 = ClusterGroup(comp2).resize((w, h))

    ratio1 = chamfer_ratio(cluster, cluster1, (0, 0))
    ratio2 = chamfer_ratio(cluster, cluster2, (0, 0))

    print(ratio1, ratio2)
    

if __name__ == "__main__":
    test4()