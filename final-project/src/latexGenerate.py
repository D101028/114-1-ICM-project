import hashlib
import os
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from PIL import Image

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageChops

def encode_macro(text: str) -> str:
    """
    將字串轉換為檔案系統安全的名稱，並在末尾以 `_{hash}` 添加防大小寫碰撞。\\
    使用 UTF-8 編碼並對特殊符號進行百分比轉義。
    """
    h = hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
    encoded = urllib.parse.quote(text, safe="") + "_" + h
    return encoded

def decode_macro(encoded_text: str) -> str:
    """
    將編碼後的檔案名稱還原回原始字串。不包含末尾的 hash
    """
    return urllib.parse.unquote("_".join(encoded_text.split("_")[:-1]))

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

def crop_img_obj(img: Image.Image, bg_color: tuple[int, int, int] | None = (255, 255, 255)):
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

def create_img(macro: str, save_dir: str):
    name = f"{encode_macro(macro)}.png"
    fp = f"{save_dir}/{name}"
    try:
        latex_symbol_to_png(
            macro, 
            out_path = fp, 
            background = (255,255,255)
        )
    except Exception as e:
        print(f"Error occurred when converting {macro}: \n\n {e}")
        if input("continue? [y/N]: ").lower() != "y":
            quit()
        return 
    img = ImageEnhance.Contrast(Image.open(fp).convert("L")).enhance(4.0)
    crop_img_obj(img, (255,255,255)).convert("L").save(fp)
    print(macro, "converted")

def test(save_dir = "./templates"):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    samples = r"""
0 1 2 3 4 5 6 7 8 9
a b c d e f g h i j k l m n o p q r s t u v w x y z
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
\text{a} \text{b} \text{c} \text{d} \text{e} \text{f} \text{g} \text{h} \text{i} \text{j} 
\text{k} \text{l} \text{m} \text{n} \text{o} \text{p} \text{q} \text{r} \text{s} \text{t} 
\text{u} \text{v} \text{w} \text{x} \text{y} \text{z}
\text{A} \text{B} \text{C} \text{D} \text{E} \text{F} \text{G} \text{H} \text{I} \text{J} 
\text{K} \text{L} \text{M} \text{N} \text{O} \text{P} \text{Q} \text{R} \text{S} \text{T} 
\text{U} \text{V} \text{W} \text{X} \text{Y} \text{Z}
+ - * / | \| ( ) = [ ] \{ \} , \lfloor \rfloor \lceil \rceil 
\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \vartheta \iota 
\kappa \lambda \mu \nu \xi \pi \varpi \rho \varrho 
\sigma \varsigma \tau \upsilon \phi \varphi \chi \psi \omega 
\Gamma \varGamma \Delta \varDelta \Theta \varTheta \Lambda \varLambda \Xi \varXi \Pi \varPi 
\Sigma \varSigma \Upsilon \varUpsilon \Phi \varPhi \Psi \varPsi \Omega \varOmega 
\aleph \beth \daleth \gimel
\sum \prod \coprod \int \oint \iint \bigcap \bigcup \bigoplus \bigotimes \bigodot \bigsqcup 
\ast \star \cdot \circ \diamond \times \div \circledast \circledcirc 
\circleddash \pm \mp \amalg \odot \oplus \otimes \Box \boxplus \cap \cup 
\sqcap \sqcup \wedge \vee \dagger \ddagger \intercal \lhd \rhd 
\unlhd \unrhd \bigtriangledown \bigtriangleup \equiv \cong \neq \sim \simeq \approx 
\propto \models \leq \geq \prec \succ \preceq \succeq \ll \gg \subset \supset \subseteq \supseteq 
\perp \parallel \in \ni \notin \triangleq \doteqdot \thickapprox \fallingdotseq \risingdotseq 
\therefore \because \leqq \geqq \leqslant \geqslant \lessapprox \gtrapprox \subseteqq \supseteqq 
\preccurlyeq \succcurlyeq \Vdash \ncong \nparallel \ntriangleleft \ntrianglelefteq \ntriangleright 
\ntrianglerighteq \nleq \ngeq \nleqq \ngeqq \nleqslant \ngeqslant \nless \ngtr \nprec \nsucc \npreceq \nsucceq 
\lneq \gneq \lneqq \gneqq \nsubseteq \nsupseteq \nsubseteqq \nsupseteqq \subsetneq \supsetneq 
\subsetneqq \supsetneqq \leftarrow \longleftarrow \Leftarrow \Longleftarrow \rightarrow \longrightarrow 
\Rightarrow \Longrightarrow \leftrightarrow \longleftrightarrow \Leftrightarrow \Longleftrightarrow 
\uparrow \Uparrow \downarrow \Downarrow \updownarrow \Updownarrow \nleftarrow \nrightarrow 
\nLeftarrow \nRightarrow \nleftrightarrow \nLeftrightarrow \mapsto \longmapsto 
\hookleftarrow \hookrightarrow \leftharpoonup \rightharpoonup \leftharpoondown \rightharpoondown 
\rightleftharpoons \leftrightharpoons \upharpoonleft \downharpoonleft \upharpoonright \downharpoonright 
\nearrow \searrow \swarrow \nwarrow \leftleftarrows \rightrightarrows \leftrightarrows \rightleftarrows 
\upuparrows \downdownarrows \curvearrowleft \curvearrowright 
\infty \nabla \partial \eth \clubsuit \diamondsuit \heartsuit \spadesuit 
\Im \Re \forall \exists \nexists \emptyset \varnothing \ell \bigstar \hbar \hslash \mho 
\wp \angle \measuredangle \sphericalangle \complement \blacksquare 
\mathcal{A} \mathcal{B} \mathcal{C} \mathcal{D} \mathcal{E} \mathcal{F} \mathcal{G} \mathcal{H} 
\mathcal{I} \mathcal{J} \mathcal{K} \mathcal{L} \mathcal{M} \mathcal{N} \mathcal{O} \mathcal{P} 
\mathcal{Q} \mathcal{R} \mathcal{S} \mathcal{T} \mathcal{U} \mathcal{V} \mathcal{W} \mathcal{X} 
\mathcal{Y} \mathcal{Z} 
\mathbb{A} \mathbb{B} \mathbb{C} \mathbb{D} \mathbb{E} \mathbb{F} \mathbb{G} \mathbb{H} 
\mathbb{I} \mathbb{J} \mathbb{K} \mathbb{L} \mathbb{M} \mathbb{N} \mathbb{O} \mathbb{P} 
\mathbb{Q} \mathbb{R} \mathbb{S} \mathbb{T} \mathbb{U} \mathbb{V} \mathbb{W} \mathbb{X} 
\mathbb{Y} \mathbb{Z} 
\mathfrak{A} \mathfrak{B} \mathfrak{C} \mathfrak{D} \mathfrak{E} \mathfrak{F} \mathfrak{G} 
\mathfrak{H} \mathfrak{I} \mathfrak{J} \mathfrak{K} \mathfrak{L} \mathfrak{M} \mathfrak{N} 
\mathfrak{O} \mathfrak{P} \mathfrak{Q} \mathfrak{R} \mathfrak{S} \mathfrak{T} \mathfrak{U} 
\mathfrak{V} \mathfrak{W} \mathfrak{X} \mathfrak{Y} \mathfrak{Z} 
""".strip().replace("\n", " ")
    # ignore \hat \dot \check \ddot \tilde \breve \acute \bar \grave \vec \sqrt \frac
    
    tex_list = samples.split()
    for tex in tex_list:
        create_img(tex, save_dir)

if __name__ == "__main__":
    test()
