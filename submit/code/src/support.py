from PIL import Image
import random

def add_noise(image_path, output_path, noise_level=0.05):
    """
    在圖片上添加隨機噪點
    :param image_path: 原始圖片路徑
    :param output_path: 輸出圖片路徑
    :param noise_level: 噪點比例 (0~1)，例如 0.05 表示 5% 的像素會被添加噪點
    """
    # 開啟圖片
    img = Image.open(image_path).convert("RGB")
    pixels = img.load()

    width, height = img.size
    num_noise_pixels = int(width * height * noise_level)

    for _ in range(num_noise_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        # 隨機生成一個顏色噪點
        noise_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        pixels[x, y] = noise_color # type: ignore

    # 儲存結果
    img.save(output_path)
    print(f"已生成帶噪點圖片：{output_path}")

if __name__ == "__main__":
    # 範例使用
    add_noise("testcases/case02.png", "testcases/case04.png", noise_level=0.01)



