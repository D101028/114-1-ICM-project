import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def crop_img(input_path: str, 
             bg_color: tuple[int, int, int] | None = (255,255,255), 
             save_bg_color: tuple[int, int, int] | None = None):
    """
    將圖片裁切到非透明（或非背景）像素的最外邊界，並可指定輸出背景顏色。
    - bg_color: 無 alpha 圖時用來辨識背景（例如 (255,255,255)），None 則取左上角像素。
    - save_bg_color: 輸出時的背景顏色；若為 None 則保留透明（若有 alpha）。
                     例如 (255,255,255) 轉為白底、(0,0,0) 轉為黑底。
    """
    img = Image.open(input_path)

    # 有 alpha：用 alpha 決定裁切範圍
    if bg_color is None and (img.mode in ("RGBA", "LA") or ("transparency" in img.info)):
        if img.mode not in ("RGBA", "LA"):
            img = img.convert("RGBA")
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        cropped = img.crop(bbox) if bbox else img

        # 決定輸出背景
        if save_bg_color is None:
            # 保留透明
            out = cropped
        else:
            # 貼到純色背景
            bg = Image.new("RGB", cropped.size, save_bg_color)
            out = Image.alpha_composite(bg.convert("RGBA"), cropped).convert("RGB")

    else:
        # 無 alpha：用背景顏色比對差異找內容範圍
        rgb = img.convert("RGB")
        if bg_color is None:
            bg_color = rgb.getpixel((0, 0))
        bg = Image.new("RGB", rgb.size, bg_color)
        diff = ImageChops.difference(rgb, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        cropped = img.crop(bbox) if bbox else img

        # 無 alpha 圖直接輸出；若指定 save_bg_color，轉貼到該底色（通常不需要）
        if save_bg_color is None:
            out = cropped.convert("RGB")
        else:
            canvas = Image.new("RGB", cropped.size, save_bg_color)
            canvas.paste(cropped)
            out = canvas

    return out

def _crop_img_obj(img: Image.Image, bg_color: tuple[int, int, int] | None = (255, 255, 255)):
    # Use alpha if requested/available
    if bg_color is None and (img.mode in ("RGBA", "LA") or ("transparency" in img.info)):
        if img.mode not in ("RGBA", "LA"):
            img = img.convert("RGBA")
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        return img.crop(bbox) if bbox else img
    else:
        rgb = img.convert("RGB")
        use_bg = bg_color if bg_color is not None else rgb.getpixel((0, 0))
        bg_img = Image.new("RGB", rgb.size, use_bg)
        diff = ImageChops.difference(rgb, bg_img)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        return img.crop(bbox) if bbox else img

def dist_compare(img1: Image.Image, img2: Image.Image, 
                 bg: tuple[int, int, int] | None = (255, 255, 255), 
                 to_crop: bool = False) -> float:
    """
    Crop non-background regions from img1 and img2 (if bg is None, prefer transparency as background),
    resize img1 or img2 size, and return the similarity ratio [0, 1] between the two images' pixel vectors.
    """
    if to_crop:
        img1 = _crop_img_obj(img1, bg)
        img2 = _crop_img_obj(img2, bg)

    # Resize target to source size
    if img1.size < img2.size:
        img2 = img2.resize(img1.size)
    elif img1.size > img2.size:
        img1 = img1.resize(img2.size)

    # Coerce both to the same mode (RGBA) and compute L2 on flattened arrays
    a = img1.convert("RGBA")
    b = img2.convert("RGBA")

    arr1 = np.asarray(a, dtype=np.float32).ravel()
    arr2 = np.asarray(b, dtype=np.float32).ravel()

    l2_distance = np.linalg.norm(arr1 - arr2, ord=2)
    max_distance = np.sqrt(len(arr1)) * 255  # Maximum possible distance

    similarity = 1 - (l2_distance / max_distance)
    return float(np.clip(similarity, 0, 1)) # ensure the value is in [0, 1]

if __name__ == "__main__":
    img1 = crop_img('data/in3.png')
    img2 = crop_img('data/in1.png')

    import time 
    start = time.time()
    for _ in range(100):
        dist_compare(img1, img2, (255,255,255))
    
    print(time.time() - start)
    
