import os
from typing import List

from PIL import Image

from .cluster import ClusterGroup, extract_components_from_pil
from .latexGenerate import decode_macro

class TexMacro:
    def __init__(self, macro: str, image: Image.Image) -> None:
        """
        Docstring for __init__
        
        :param self: 
        :param macro: Tex Macro in string (e.g. r"\\alpha")
        :type macro: str
        :param img: Cropped PIL image of the Tex Macro
        :type img: Image.Image
        """

        self.macro = macro
        self.image = image.convert("L")  # saved in gray mode

        self.components, self.bbox_yxhw, _ = extract_components_from_pil(image)
        self.yx_ratio = self.bbox_yxhw[2] / self.bbox_yxhw[3]

        self.cluster = ClusterGroup(self.components)

def load_tex_macros(folder: str) -> List[TexMacro]:
    out: List[TexMacro] = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            macro = ".".join(file.split(".")[:-1])
            image = Image.open(os.path.join(root, file)).convert("L")
            out.append(TexMacro(decode_macro(macro), image))
    
    return out
