import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance, ImageDraw
from prettytable import PrettyTable
from typing import List, Tuple

from macrotopng import latex_symbol_to_png
from mc import level_mark_components_and_clusters_pil as lmccp2
from mc import ClusterGroup, extract_components_from_pil
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
        name = macro.replace("\\", "_bs_") + "_" + str(random.randint(0, 100)) + ".png"
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
    create_img(":")
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

def adaptive_cluster(
        img: Image.Image, sauce: dict[AnsType, Image.Image], accept_sim = 0.7, max_depth = 16
        ) -> AdaptiveReturnType:

    def recurse(img: Image.Image, depth = 0, fix_ratio_x = 1.0, fix_ratio_y = 1.0, topleft = (0, 0)) -> AdaptiveReturnType:
        output_pil, components, clusters, boxes, black_height = lmccp2(
            img, 2 + 0.1 * fix_ratio_x, 0.1 + 0.1 * fix_ratio_y, 4
        )
        if len(boxes) == 1 and depth < max_depth:
            return recurse(img, depth + 1, fix_ratio_x + 0.5, fix_ratio_y + 2, topleft)
        
        rec_out: AdaptiveReturnType = []
        for cluster, box in zip(clusters, boxes):
            y, x, h, w = box
            curr_topleft = (topleft[0] + , topleft[1] + )
            r = w / h
            tgt = cluster.to_L()
            if len(cluster.components) > 4 and depth < max_depth:
                rec_out.extend(recurse(tgt, depth+1, topleft=curr_topleft))
                continue
            ans = None
            max_sim = 0.0
            for f, src in sauce.items():
                # 極端長寬比排除
                r2 = src.size[1] / src.size[0]
                if (not (0.1 < r < 10) or not (0.1 < r2 < 10)) and not (0.1 < r / r2 < 10):
                    continue

                curr = dist_compare(src, tgt, (255,255,255))
                if max_sim < curr:
                    ans = f
                    max_sim = curr
            if max_sim < accept_sim and depth < max_depth and len(cluster.components) > 1:
                rec_out.extend(recurse(tgt, depth+1, topleft=curr_topleft))
                continue
            if max_sim < accept_sim and ans is not None:
                print("doubted ans: ", ans, max_sim)
            rec_out.append((cluster, ans, max_sim, curr_topleft))
        return rec_out
    
    return recurse(img)

def test4():
    MAX_DEPTH = 16

    sauce: dict[AnsType, Image.Image] = {}
    for root, dirs, files in os.walk("./templates"):
        for f in files:
            path = f"{root}/{f}"
            src = Image.open(path)
            sauce[f] = src
    
    src_img = Image.open("data/in0.png")
    src_img = src_img.convert("L")
    enhancer = ImageEnhance.Contrast(src_img)
    src_img = enhancer.enhance(2.0)

    import time 
    start = time.time()
    out = adaptive_cluster(src_img, sauce)
    print(time.time() - start)

    myTable = PrettyTable(["Position (y, x, h, w)", "Centroid (x, y)", "Answer", "Similarity"])
    tableRows = []
    for cluster_gp, ans, sim, topleft in out:
        dy, dx = topleft
        y, x, h, w = cluster_gp.get_bbox_yxhw()
        cx, cy = cluster_gp.get_centroid()
        tableRows.append([
            (y+dy, x+dx, h, w), 
            (cx+dx, cy+dy), 
            ans, sim
        ])
    tableRows.sort(
        key = lambda row: row[1][0]
    )
    for row in tableRows:
        myTable.add_row(row)
    src_img.save("test.png")
    print(myTable)

def test1():
    img: Image.Image = Image.open("data/in1.png")
    img.convert("L")
    components, _, _ = extract_components_from_pil(img)
    cluster = ClusterGroup(components[:5])
    print(cluster.get_centroid())
    

if __name__ == "__main__":
    test4()