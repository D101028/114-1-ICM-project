import numpy as np
import cv2
import os
from PIL import Image
from prettytable import PrettyTable

from macrotopng import latex_symbol_to_png
from mc import level_mark_components_and_clusters_pil
from pixelwise import dist_compare

def test0():
    texs = [
        r"e^{i\pi^{jk^{ij}}}", 
        r"\text{Apple, I have icecream } \sum_{i=0}^{2^{10}}e^{-i}", 
        r"j \cdot \sum^{i}", 
        r"\Theta \varTheta \Xi \varXi \| \Vert \lim_{x \to \infty} \sqrt[n]{abc}\circledast \circledcirc \div \doublebarwedge \doteqdot \fallingdotseq \risingdotseq \Vvdash ", 
        r"\text{Apple, I have icecream } ", 
        r"i\quad j\quad := \doteqdot \div \doteq "
    ]

    for idx, tex in enumerate(texs):
        latex_symbol_to_png(
            tex, 
            dpi=600, 
            out_path=f"./data/in{idx}.png", 
            background=(255,255,255)
        )

def test():
    for i in range(0, 6):
        img = Image.open(f"./data/in{i}.png")
        out_img = level_mark_components_and_clusters_pil(
            img, wx=4, wy=0, 
            refine_min_size=4, refine_ratio=0.3, diag_ratio=-1)[0]
        out_img.save(f"./data/out{i}.png")

def test2():
    img = Image.open("test3.png")

    # # convert
    # img_arr = np.array(img.convert("L"))
    # cv2_img = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
    # # 濾波（可選 GaussianBlur 或 medianBlur）
    # blurred = cv2.medianBlur(cv2_img, (5, 5), 0)
    # # 二值化
    # _, binary = cv2.threshold(blurred, 191, 255, cv2.THRESH_BINARY)

    # img = Image.fromarray(binary)

    # img.save("tmp.png")

    out_img = level_mark_components_and_clusters_pil(img, 4, 0.6, 4, 0.5, 1/4)[0]
    out_img.save(f"out.png")

def test3():
    samples = [
        (r"\sum", "sum.png"), 
        (r"j", "j.png"), 
        (r"i", "i.png"), 
        (r"\cdot", "cdot.png"), 
    ]
    for macro, name in samples:
        latex_symbol_to_png(
            macro, 
            out_path = f"templates/{name}", 
            background = (255,255,255)
        )

def test4():
    MAX_DEPTH = 7
    # crop_func = lambda src_img: level_mark_components_and_clusters_pil(
    #     src_img, 4, 0.3, float('inf'), diag_ratio=-1
    # )
    myTable = PrettyTable(["y, x, h, w", "Answer", "Similarity", "Depth"])
    pos_info: list[tuple[int, int]] = [] # list[(Num, Depth), ...]
    def inner(img: Image.Image, depth = 0):
        output_pil, centroids, clusters, boxes, black_height = level_mark_components_and_clusters_pil(
            img, 4, 0.3 + 0.1 * depth, float('inf'), diag_ratio=-1
        )
        output_pil.save(f"./data/p_{depth}.png")
        
        num = 0
        for box in boxes:
            y, x, h, w = box
            tgt = img.crop((x, y, x + w, y + h))
            # tgt.save(f"data/{depth}.png")
            ans = None
            max_sim = 0
            for root, dirs, files in os.walk("./templates"):
                for f in files:
                    path = f"{root}/{f}"
                    src = Image.open(path)
                    curr = dist_compare(src, tgt, (255,255,255))
                    if max_sim < curr:
                        ans = f
                        max_sim = curr
            if max_sim < 0.7 and depth < MAX_DEPTH:
                inner(tgt, depth + 1)
            else:
                myTable.add_row([box, ans, max_sim, depth])
            num += 1
    src_img = Image.open("data/in2.png")
    src_img.convert("LA")
    inner(src_img)
    print(myTable)

if __name__ == "__main__":
    test4()