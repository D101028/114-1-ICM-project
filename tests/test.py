import numpy as np
import cv2
import os
import sys
from PIL import Image, ImageEnhance, ImageDraw
from prettytable import PrettyTable
from typing import List, Tuple, Dict
from scipy.spatial import KDTree

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

def chamfer_ratio(cluster1: ClusterGroup, cluster2: ClusterGroup, topleft: Tuple[int, int]):
    arr1 = np.empty((0, 2), dtype=int)
    arr2 = np.empty((0, 2), dtype=int)

    for comp in cluster1.components:
        pts = comp.pixels_below_threshold(127)   # shape (K,2)
        pts[:, 0] = pts[:, 0] - topleft[0]
        pts[:, 1] = pts[:, 1] - topleft[1]
        if pts.size == 0:
            continue
        arr1 = np.vstack([arr1, pts])

    for comp in cluster2.components:
        pts = comp.pixels_below_threshold(127)
        if pts.size == 0:
            continue
        arr2 = np.vstack([arr2, pts])
    chamfer_dist = chamfer_distance(arr1, arr2)
    y, x, h, w = cluster1.get_bbox_yxhw()
    ratio = chamfer_dist / (2 * ((w-1)**2+(h-1)**2)**0.5)


    img1 = cluster1.to_L().convert("RGB")
    for pt in arr1:
        try:
            img1.putpixel([pt[1], pt[0]], (0, 255, 0))
        except:
            pass 
    
    # import random
    # img1.save(f"test_{random.randint(0, 1000)}.png")

    return 1 - ratio

def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    計算兩個點集 A, B 的 Hausdorff 距離
    H(A, B) = max(h(A,B), h(B,A))
    h(A,B) = max_{a in A} min_{b in B} ||a-b||_2
    """
    if A.size == 0 or B.size == 0:
        raise ValueError("點集不能為空")

    # 對每個 a in A，計算到 B 的最短距離
    dists_A_to_B = np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2), axis=1)
    h_A_B = dists_A_to_B.max()

    # 對每個 b in B，計算到 A 的最短距離
    dists_B_to_A = np.min(np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2), axis=1)
    h_B_A = dists_B_to_A.max()

    return max(h_A_B, h_B_A)

def hausdorff_ratio(cluster1: ClusterGroup, cluster2: ClusterGroup, topleft: Tuple[int, int]):
    arr1 = np.empty((0, 2), dtype=int)
    arr2 = np.empty((0, 2), dtype=int)

    for comp in cluster1.components:
        pts = comp.get_soft_contour_points()   # shape (K,2)
        pts[:, 0] = pts[:, 0] - topleft[0]
        pts[:, 1] = pts[:, 1] - topleft[1]
        if pts.size == 0:
            continue
        arr1 = np.vstack([arr1, pts])

    for comp in cluster2.components:
        pts = comp.get_soft_contour_points()
        if pts.size == 0:
            continue
        arr2 = np.vstack([arr2, pts])
    dist = hausdorff_distance(arr1, arr2)
    y, x, h, w = cluster1.get_bbox_yxhw()
    ratio = dist / (2 * ((w-1)**2+(h-1)**2)**0.5)


    img1 = cluster1.to_L().convert("RGB")
    for pt in arr1:
        try:
            img1.putpixel([pt[1], pt[0]], (0, 255, 0))
        except:
            pass 
    
    # import random
    # img1.save(f"test_{random.randint(0, 1000)}.png")

    return 1 - ratio

def adaptive_cluster(
        img: Image.Image, sauce: SauceType, accept_sim = 0.1, max_depth = 16
    ) -> AdaptiveReturnType:

    def recurse(img: Image.Image, depth = 0, fix_ratio_x = 1.0, fix_ratio_y = 1.0, topleft = (0, 0)) -> AdaptiveReturnType:
        wx = 2 + 0.1 * fix_ratio_x
        wy = 0.1 + 0.1 * fix_ratio_y
        refine_min_size = 4
        refine_ratio = 0.5

        # extract components
        components, bbox_yxhw, gray = extract_components_from_pil(img)

        # threshold 
        base_threshold = bbox_yxhw[2] * 0.4

        # clustering
        clusters = hierarchical_cluster(
            components,
            base_threshold=base_threshold,
            wx=wx,
            wy=wy,
            refine_min_size=refine_min_size,
            refine_ratio=refine_ratio,
            build_cluster_masks=False,
            image_shape=gray.shape # type: ignore
        )

        if len(clusters) == 1 and len(clusters[0].components) > 1 and depth < max_depth:
            return recurse(img, depth + 1, fix_ratio_x + 0.5, fix_ratio_y + 2, topleft)
        
        rec_out: AdaptiveReturnType = []
        for cluster in clusters:
            y, x, h, w = cluster.get_bbox_yxhw()
            next_topleft = (topleft[0] + y, topleft[1] + x)
            r = h / w
            tgt = cluster.to_L()
            if len(cluster.components) > 4 and depth < max_depth:
                rec_out.extend(recurse(tgt, depth+1, topleft=next_topleft))
                continue
            ans = None
            max_sim = 0.0
            if r < 0.1:
                for f, (src, yx_ratio, cluster2) in sauce.items():
                    if yx_ratio > 0.125:
                        continue
                    # curr = dist_compare(src, tgt, (255,255,255))
                    curr = hausdorff_ratio(cluster, cluster2, next_topleft)
                    
                    if max_sim < curr:
                        ans = f
                        max_sim = curr
            elif r > 10:
                for f, (src, yx_ratio, cluster2) in sauce.items():
                    if yx_ratio < 8:
                        continue

                    # curr = dist_compare(src, tgt, (255,255,255))
                    curr = hausdorff_ratio(cluster, cluster2, next_topleft)
                    if max_sim < curr:
                        ans = f
                        max_sim = curr
            else:
                for f, (src, yx_ratio, cluster2) in sauce.items():
                    if yx_ratio > 10 or yx_ratio < 0.1:
                        continue
                    # curr = dist_compare(src, tgt, (255,255,255))
                    curr = hausdorff_ratio(cluster, cluster2, next_topleft)
                    if max_sim < curr:
                        ans = f
                        max_sim = curr
            if max_sim < accept_sim and depth < max_depth and len(cluster.components) > 1:
                rec_out.extend(recurse(tgt, depth+1, topleft=next_topleft))
                continue
            if max_sim < accept_sim and ans is not None:
                print("doubted ans: ", ans, max_sim)
            rec_out.append((cluster, ans, max_sim, topleft))
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
    img1 = Image.open("templates/2_14.png")
    # img2 = Image.open("templates/3_92.png")
    from PIL import ImageOps
    img2 = ImageOps.invert(img1.convert("L"))
    img2.save("test.png")
    components1, _, _ = extract_components_from_pil(img1)
    components2, _, _ = extract_components_from_pil(img2)

    cluster1 = ClusterGroup(components1)
    cluster2 = ClusterGroup(components2)
    
    ratio = chamfer_ratio(cluster1, cluster2, (0, 0))
    
    print(ratio)

    ratio0 = chamfer_ratio(cluster1, cluster1, (0, 0))
    print(ratio0)
    

if __name__ == "__main__":
    test4()