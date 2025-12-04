import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
from prettytable import PrettyTable

from macrotopng import latex_symbol_to_png
from mc import level_mark_components_and_clusters_pil as lmccp2
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

def test4():
    MAX_DEPTH = 16
    myTable = PrettyTable(["y, x, h, w", "Answer", "Similarity", "Depth"])
    
    sauce: dict[str, Image.Image] = {}
    for root, dirs, files in os.walk("./templates"):
        for f in files:
            path = f"{root}/{f}"
            src = Image.open(path)
            sauce[f] = src
    
    def inner(img: Image.Image, depth = 0, fix_ratio_x = 1.0, fix_ratio_y = 1.0) -> None:
        output_pil, components, clusters, boxes, black_height = lmccp2(
            img, 2 + 0.1 * fix_ratio_x, 0.1 + 0.1 * fix_ratio_y, 4
        )
        import random
        if len(boxes) == 1 and depth < MAX_DEPTH:
            return inner(img, depth + 1, fix_ratio_x + 0.5, fix_ratio_y + 2)
        output_pil.save(f"./data/p_{depth}_{random.randint(0, 1000)}.png")
        
        for cluster, box in zip(clusters, boxes):
            y, x, h, w = box
            r = w / h
            tgt = cluster.to_L()
            if len(cluster.components) > 4 and depth < MAX_DEPTH:
                inner(tgt, depth+1)
                continue
            ans = None
            max_sim = 0
            for f, src in sauce.items():
                # 極端長寬比排除
                r2 = src.size[1] / src.size[0]
                if (not (0.1 < r < 10) or not (0.1 < r2 < 10)) and not (0.1 < r / r2 < 10):
                    continue

                curr = dist_compare(src, tgt, (255,255,255))
                if max_sim < curr:
                    ans = f
                    max_sim = curr
            if max_sim < 0.7 and depth < MAX_DEPTH and len(cluster.components) > 1:
                inner(tgt, depth + 1)
                continue
            if max_sim < 0.7 and ans is not None:
                fp = f"data/q_{random.randint(0, 100)}"
                tgt.save(f"{fp}.png")
                Image.open(f"templates/{ans}").convert("L").resize(tgt.size).save(f"{fp}-1.png")
                print(fp, ans)
            myTable.add_row([box, ans, max_sim, depth])

    src_img = Image.open("data/in1.png")
    src_img = src_img.convert("L")
    enhancer = ImageEnhance.Contrast(src_img)
    src_img = enhancer.enhance(2.0)
    import time 
    start = time.time()
    inner(src_img)
    print(time.time() - start)
    print(myTable)

if __name__ == "__main__":
    test4()