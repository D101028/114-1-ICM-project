import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def render_symbol_to_png(symbol, filename, dpi=300, pad=0.25, transparent=True):
    """
    使用 xelatex 將單一 LaTeX 符號渲染為 PNG。
    """
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.rcfonts": False,
        "text.latex.preamble": r"\usepackage{amsmath,amsthm,amssymb, mathtools}",
        "pgf.texsystem": "xelatex",
    })

    fig = plt.figure(figsize=(2, 2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, 0.5, f"${symbol}$", ha="center", va="center", fontsize=32)

    fig.savefig(filename, dpi=dpi, transparent=transparent,
                bbox_inches="tight", pad_inches=pad)
    plt.close(fig)

# def render_symbol_to_png(symbol, filename, dpi=200, pad=0.25, transparent=True):
#     """
#     將單一 LaTeX 符號渲染為 PNG。
#     symbol: 不含 $$ 的 LaTeX 內容（例如 r'\alpha' 或 r'\mathbb{R}'）
#     filename: 輸出檔名（包含路徑）
#     dpi: 圖片解析度
#     pad: 圖像周圍留白比例（inches）
#     transparent: 是否透明背景
#     """
#     fig = plt.figure(figsize=(2, 2), dpi=dpi)
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.axis('off')

#     # 使用 mathtext inline 格式：$...$
#     text = f"${symbol}$"

#     # 將文字置中
#     ax.text(0.5, 0.5, text, ha='center', va='center')

#     # 自動根據文字調整畫布大小
#     fig.canvas.draw()
#     # 使用 tight_layout=False，改用 bbox_inches='tight' + pad_inches 控制留白
#     fig.savefig(filename, dpi=dpi, transparent=False, bbox_inches='tight', pad_inches=pad, facecolor='white', edgecolor='white')
#     plt.close(fig)

if __name__ == "__main__":

    # 使用 matplotlib 的內建 mathtext，避免依賴系統 LaTeX
    plt.rcParams['mathtext.fontset'] = 'stix'   # 也可改為 'dejavusans' 或 'cm'
    plt.rcParams['font.size'] = 72              # 控制輸出字體大小

    # 想要輸出的 LaTeX 符號清單（不需加 $$，但需用 \ 指令）
    symbols = r"""
a b c d e f g h i j k l m n o p q r s t u v w x y z
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
""".strip().replace('\n', '').split()

    # 輸出資料夾
    out_dir = "./latex_symbols"
    os.makedirs(out_dir, exist_ok=True)

    # 產生檔名的安全函數（移除反斜線與特殊字元）
    def safe_name(symbol):
        name = symbol.replace('\\', '').replace('{', '').replace('}', '')
        name = name.replace('^', 'sup').replace('_', 'sub')
        name = name.replace('/', '_over_')
        # 避免空字串
        return name if name.strip() else "symbol"

    # 主迴圈
    for sym in symbols:
        fname = os.path.join(out_dir, f"{safe_name(sym)}.png")
        try:
            render_symbol_to_png(sym, fname, dpi=300, pad=0.3, transparent=False)
        except:
            pass 

    print(f"Done. Images saved to: {os.path.abspath(out_dir)}")
