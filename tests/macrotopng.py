import subprocess
# import tempfile
# from pathlib import Path
import fitz  # PyMuPDF
import os

TEMP_DIR = "."
if not os.path.isdir(TEMP_DIR):
    os.mkdir(TEMP_DIR)

def latex_symbol_to_png(macro, compiler="xelatex", dpi=300):
    tex_path = os.path.join(TEMP_DIR, "symbol.tex")
    pdf_path = os.path.join(TEMP_DIR, "symbol.pdf")
    cropped_pdf_path = os.path.join(TEMP_DIR, "symbol-crop.pdf")
    png_path = os.path.join(TEMP_DIR, "symbol.png")
    to_clean = (tex_path, pdf_path, cropped_pdf_path, 
                os.path.join(TEMP_DIR, "symbol.aux"), os.path.join(TEMP_DIR, "symbol.log"))

    # --- Step 1: Write TeX file ---
    with open(tex_path, mode="w") as file:
        file.write(rf"""
\documentclass[preview,border=2pt]{{standalone}}
\usepackage{{amsmath, amssymb}}
\begin{{document}}
\[
{macro}
\]
\end{{document}}
""")

    # --- Step 2: compile ---
    subprocess.run([compiler, "-interaction=nonstopmode", str(tex_path)],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # --- Step 3: crop to bounding box ---
    subprocess.run(["pdfcrop", pdf_path, cropped_pdf_path],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # --- Step 4: convert PDF → PNG with PyMuPDF ---
    pdf = fitz.open(cropped_pdf_path)
    page = pdf[0]

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=True)

    pix.save(png_path)

    pdf.close()

    # clean files
    for path in to_clean:
        os.remove(path)

    return str(png_path), pix.width, pix.height

# def latex_symbol_to_png(
#     macro,
#     compiler="xelatex",
#     dpi=300,
# ):
#     """
#     Convert a LaTeX math symbol macro like '\\alpha' into a PNG image
#     using xelatex or lualatex.
#     Returns: (png_path, width_px, height_px)
#     """

#     assert compiler in ("xelatex", "lualatex"), "compiler must be xelatex or lualatex"

#     with tempfile.TemporaryDirectory() as tmpdir:
#         tmpdir = Path(tmpdir)
#         tex_path = tmpdir / "symbol.tex"
#         pdf_path = tmpdir / "symbol.pdf"
#         cropped_pdf_path = tmpdir / "symbol-crop.pdf"
#         png_path = tmpdir / "symbol.png"

#         # --- Step 1: Write the latex file ---
#         tex_content = rf"""
# \documentclass[preview,border=2pt]{{standalone}}
# \usepackage{{amsmath}}
# \usepackage{{amssymb}}
# \begin{{document}}
# \[
#     {macro}
# \]
# \end{{document}}
# """
#         tex_path.write_text(tex_content)

#         # --- Step 2: Compile to PDF ---
#         subprocess.run(
#             [
#                 compiler,
#                 "-interaction=nonstopmode",
#                 str(tex_path)
#             ],
#             cwd=tmpdir,
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )

#         # --- Step 3: Crop PDF to bounding box ---
#         subprocess.run(
#             ["pdfcrop", pdf_path, cropped_pdf_path],
#             cwd=tmpdir,
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )

#         # --- Step 4: Convert PDF to PNG with bounding box preserved ---
#         subprocess.run(
#             [
#                 "gs",
#                 "-dSAFER",
#                 "-dBATCH",
#                 "-dNOPAUSE",
#                 "-sDEVICE=pngalpha",
#                 f"-r{dpi}",
#                 f"-sOutputFile={png_path}",
#                 str(cropped_pdf_path),
#             ],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )

#         # --- Step 5: Get image size ---
#         from PIL import Image
#         img = Image.open(png_path)
#         w, h = img.size

#         return str(png_path), w, h

png_path, width, height = latex_symbol_to_png(r"\sum", compiler="xelatex")
print("output:", png_path, width, height)
