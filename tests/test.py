import numpy as np
import cv2
from PIL import Image

from macrotopng import latex_symbol_to_png
from mc import level_mark_components_and_clusters_pil

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

if __name__ == "__main__":
    test0()
    test()