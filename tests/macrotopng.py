import subprocess
import tempfile
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

def latex_symbol_to_png(
    macro,
    compiler="xelatex",
    dpi=300,
    out_path="symbol.png", 
    background=None
):
    """
    Convert a LaTeX math symbol macro like '\\alpha' into a PNG image
    using xelatex or lualatex.
    Returns: (png_path, width_px, height_px)
    """

    assert compiler in ("xelatex", "lualatex"), "compiler must be xelatex or lualatex"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tex_path = tmpdir / "symbol.tex"
        pdf_path = tmpdir / "symbol.pdf"
        cropped_pdf_path = tmpdir / "symbol-crop.pdf"

        # --- Step 1: Write the latex file ---
        tex_content = rf"""
\documentclass[preview,border=2pt]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\begin{{document}}
\[
    {macro}
\]
\end{{document}}
"""
        tex_path.write_text(tex_content)

        # --- Step 2: Compile to PDF ---
        subprocess.run(
            [
                compiler,
                "-interaction=nonstopmode",
                str(tex_path)
            ],
            cwd=tmpdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # --- Step 3: Crop PDF to bounding box ---
        subprocess.run(
            ["pdfcrop", pdf_path, cropped_pdf_path],
            cwd=tmpdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # --- Step 4: convert PDF → PNG with white background ---
        pdf = fitz.open(cropped_pdf_path)
        page = pdf[0]

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=True)

        if background is None:
            pix.save(out_path)
        else:
            img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            bg = Image.new("RGB", img.size, background)
            bg.paste(img, mask=img.split()[3])  # use alpha channel as mask
            bg.save(out_path, format="PNG")

        pdf.close()

        return pix.width, pix.height


if __name__ == "__main__":
    width, height = latex_symbol_to_png(
        r"\overline{ab} ijk \dot{a} = \geq \equiv \ddots e^{i\pi^{jk^{ij}}}", 
        dpi=600, 
        out_path="in1.png", 
        background=(255,255,255)
    )
    width, height = latex_symbol_to_png(
        r"\text{Apple, I have icecream } \sum_{i=0}^{2^{10}}e^{-i}", 
        dpi=600, 
        out_path="in2.png", 
        background=(255,255,255)
    )
    print("output:", width, height)
