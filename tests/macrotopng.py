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
    background=None, 
    gray_level=True
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
        with fitz.open(cropped_pdf_path) as pdf:
            page = pdf[0]

            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=True)

            if background is None:
                if gray_level:
                    # Convert pixmap to grayscale while keeping alpha
                    # pix.samples: RGBA; shape: (h*w*4)

                    # Convert pixmap → PIL image
                    img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)

                    # Split channels
                    r, g, b, a = img.split()

                    # Convert RGB → grayscale (luminosity)
                    gray = Image.merge("RGB", (r, g, b)).convert("L")

                    # Recombine gray + original alpha
                    final_img = Image.merge("LA", (gray, a))
                    final_img.save(out_path)
                else:
                    pix.save(out_path)
            else:
                img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
                if gray_level:
                    img = img.convert("LA")
                bg = Image.new("RGB", img.size, background)
                bg.paste(img, mask=img.split()[-1])  # use alpha channel as mask
                bg.save(out_path, format="PNG")

        return pix.width, pix.height

