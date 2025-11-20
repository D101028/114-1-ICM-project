import os
import numpy as np
from PIL import Image, ImageChops

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

def compare_img(img1: Image.Image, img2: Image.Image, to_size: tuple[int, int] | None = None, max_samples: int = 8192) -> float:
    """
    Resize two PIL Images to `to_size` (default 128x128), sample up to `max_samples`
    pixels (evenly distributed) and return the ratio of similarity (0.0 - 1.0).
    """
    if to_size is None:
        to_size = (128, 128)

    resample = getattr(Image, "LANCZOS", Image.BICUBIC)

    a = img1.resize(to_size, resample).convert("RGBA")
    b = img2.resize(to_size, resample).convert("RGBA")

    total = to_size[0] * to_size[1]
    if total == 0:
        return 0.0

    # sample every `step`-th pixel to keep comparisons bounded
    step = max(1, total // max_samples)

    it_a = a.getdata()
    it_b = b.getdata()

    arr_a = np.asarray(a, dtype=np.uint8).reshape(-1, 4)
    arr_b = np.asarray(b, dtype=np.uint8).reshape(-1, 4)

    indices = np.arange(0, arr_a.shape[0], step)
    sampled = indices.size

    if sampled == 0:
        same = 0.0
    else:
        # use RGB vector distance -> similarity in [0,1]
        a_rgb = arr_a[indices, :3].astype(np.float32)
        b_rgb = arr_b[indices, :3].astype(np.float32)

        diff = a_rgb - b_rgb
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance per pixel

        max_dist = 255.0 * (3 ** 0.5)
        sim = 1.0 - (dist / max_dist)
        sim = np.clip(sim, 0.0, 1.0)

        # average similarity is sum(sim) / sampled; keep numerator to match final return
        same = float(np.sum(sim))

    return (same / sampled) if sampled else 0.0

def _crop_img_obj(img: Image.Image, bg_color: tuple[int, int, int] | None):
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

def dist_compare(img1: Image.Image, img2: Image.Image, bg: tuple[int, int, int] | None = None) -> float:
    """
    Crop non-background regions from img1 and img2 (if bg is None, prefer transparency as background),
    resize img1 or img2 size, and return the similarity ratio [0, 1] between the two images' pixel vectors.
    """

    img1_c = _crop_img_obj(img1, bg)
    img2_c = _crop_img_obj(img2, bg)

    # Resize target to source size
    resample = getattr(Image, "LANCZOS", Image.BICUBIC)
    if img1_c.size > img2_c.size:
        img2_c = img2_c.resize(img1_c.size, resample)
    elif img1_c.size < img2_c.size:
        img1_c = img1_c.resize(img2_c.size, resample)

    # Coerce both to the same mode (RGBA) and compute L2 on flattened arrays
    a = img1_c.convert("RGBA")
    b = img2_c.convert("RGBA")

    arr1 = np.asarray(a, dtype=np.float32).ravel()
    arr2 = np.asarray(b, dtype=np.float32).ravel()

    l2_distance = np.linalg.norm(arr1 - arr2, ord=2)
    max_distance = np.sqrt(len(arr1)) * 255  # Maximum possible distance

    similarity = 1 - (l2_distance / max_distance)
    return float(np.clip(similarity, 0, 1))

if __name__ == "__main__":
    img1 = crop_img('./test2.png')

    for root, dirs, files in os.walk('./latex_symbols'):
        for file in files:
            img2 = crop_img(f"{root}/{file}")
            print(f"{file}: {compare_img(img1, img2)}")
        break
