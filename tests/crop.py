from PIL import Image, ImageChops
import os

def trim_to_content(input_path, output_path, bg_color=None, save_bg_color=None):
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

    out.save(output_path)


if __name__ == "__main__":
    src_dir = './latex_symbols'
    out_dir = './out_symbols'
    
    os.makedirs(out_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            trim_to_content(f"{root}/{file}", f"{out_dir}/{file}", bg_color=(255,255,255), save_bg_color=(255, 255, 255))
        break
