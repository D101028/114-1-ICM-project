import os
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

def trim_to_content(input_path, output_path, bg_color=None, save_bg_color=None):
    """
    將圖片裁切到非透明（或非背景）像素的最外邊界，並可指定輸出背景顏色。
    - bg_color: 無 alpha 圖時用來辨識背景（例如 (255,255,255)），None 則取左上角像素。
    - save_bg_color: 輸出時的背景顏色；若為 None 則保留透明（若有 alpha）。
                     例如 (255,255,255) 轉為白底、(0,0,0) 轉為黑底。
    """
    img = Image.open(input_path)

    # 有 alpha：用 alpha 決定裁切範圍
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
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

def render_symbol_to_png(symbol, filename, dpi=200, pad=0.25, transparent=True):
    """
    將單一 LaTeX 符號渲染為 PNG。
    symbol: 不含 $$ 的 LaTeX 內容（例如 r'\alpha' 或 r'\mathbb{R}'）
    filename: 輸出檔名（包含路徑）
    dpi: 圖片解析度
    pad: 圖像周圍留白比例（inches）
    transparent: 是否透明背景
    """
    fig = plt.figure(figsize=(2, 2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # 使用 mathtext inline 格式：$...$
    text = f"${symbol}$"

    # 將文字置中
    ax.text(0.5, 0.5, text, ha='center', va='center')

    # 自動根據文字調整畫布大小
    fig.canvas.draw()
    # 使用 tight_layout=False，改用 bbox_inches='tight' + pad_inches 控制留白
    fig.savefig(filename, dpi=dpi, transparent=False, bbox_inches='tight', pad_inches=pad, facecolor='white', edgecolor='white')
    plt.close(fig)

if __name__ == "__main__":

    # 使用 matplotlib 的內建 mathtext，避免依賴系統 LaTeX
    plt.rcParams['mathtext.fontset'] = 'stix'   # 也可改為 'dejavusans' 或 'cm'
    plt.rcParams['font.size'] = 72              # 控制輸出字體大小

    # 想要輸出的 LaTeX 符號清單（不需加 $$，但需用 \ 指令）
    symbols = [
        r'a',
        r'\alpha',
        r'\forall', 
        r'\nabla',
    ]

    # 輸出資料夾
    out_dir = "latex_symbols"
    os.makedirs(out_dir, exist_ok=True)

    # 產生檔名的安全函數（移除反斜線與特殊字元）
    def safe_name(symbol):
        name = symbol.replace('\\', '_').replace('{', '').replace('}', '')
        name = name.replace('^', 'sup').replace('_', 'sub')
        name = name.replace('/', '_over_')
        # 避免空字串
        return name if name.strip() else "symbol"

    # 主迴圈
    for sym in symbols:
        fname = os.path.join(out_dir, f"{safe_name(sym)}.png")
        try:
            render_symbol_to_png(sym, fname, dpi=300, pad=0.3, transparent=True)
        except:
            pass 

    print(f"Done. Images saved to: {os.path.abspath(out_dir)}")
