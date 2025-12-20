import sys
from typing import List, Tuple, Callable

from PIL import Image, ImageEnhance, ImageDraw
from prettytable import PrettyTable

from src.cluster import ClusterGroup, extract_components_from_pil, hierarchical_cluster
from src.hausdorff import hausdorff_similarity
from src.l2dist import l2_similarity
from src.template import TexMacro, load_tex_macros

class AdaptiveReturnSet:
    def __init__(self, cluster, best_macro, max_sim, second_sim, topleft, depth) -> None:
        self.cluster = cluster
        self.best_macro = best_macro
        self.max_sim = max_sim
        self.second_sim = second_sim
        self.topleft = topleft
        self.depth = depth

def adaptive_cluster(
    img: Image.Image, sauce: List[TexMacro], 
    second_sim_func: Callable[[ClusterGroup, ClusterGroup, Tuple[int, int]], float] = lambda _, __, ___: 1.0, 
    accept_sim = 0.7, second_accept_sim = 0.9, max_depth = 16
) -> List[AdaptiveReturnSet]:

    def recurse(img: Image.Image, depth = 0, fix_ratio_x = 1.0, fix_ratio_y = 1.0, topleft = (0, 0)) -> List[AdaptiveReturnSet]:
        # 1. 初始化參數與分群
        wx, wy = 2 + 0.1 * fix_ratio_x, 0.1 + 0.1 * fix_ratio_y
        components, bbox_yxhw, gray = extract_components_from_pil(img)
        
        clusters = hierarchical_cluster(
            components,
            base_threshold=bbox_yxhw[2] * 0.4,
            wx=wx, wy=wy,
            refine_min_size=4, refine_ratio=0.5,
            build_cluster_masks=False,
            image_shape=gray.shape
        )

        # 2. 特殊情況：如果分不開且還有深度，增加權重重試
        if len(clusters) == 1 and len(clusters[0].components) > 1 and depth < max_depth:
            return recurse(img, depth + 1, fix_ratio_x + 0.5, fix_ratio_y + 2, topleft)
        
        rec_out: List[AdaptiveReturnSet] = []
        for cluster in clusters:
            y, x, h, w = cluster.get_bbox_yxhw()
            next_topleft = (topleft[0] + y, topleft[1] + x)
            r = h / w
            tgt = cluster.to_L()

            # 3. 遞迴深挖：如果組件過多，強制進入下一層
            if len(cluster.components) > 4 and depth < max_depth:
                rec_out.extend(recurse(tgt, depth + 1, topleft=next_topleft))
                continue

            # 4. 統一比對邏輯
            best_macro, max_sim, second_sim = None, 0.0, 0.0
            
            # for f, (src, yx_ratio, base_cluster) in sauce.items():
            for tex_macro in sauce:
                # 比率篩選邏輯
                is_compatible = (
                    (r < 0.1 and tex_macro.yx_ratio <= 0.125) or
                    (r > 10 and tex_macro.yx_ratio >= 8) or
                    (0.1 <= r <= 10 and 0.1 <= tex_macro.yx_ratio <= 10)
                )
                if not is_compatible: continue

                l2_sim = l2_similarity(tex_macro.image, tgt)
                
                if l2_sim > max_sim:
                    max_sim = l2_sim
                    second_sim = second_sim_func(tex_macro.cluster, cluster, topleft)
                    best_macro = tex_macro

            # 5. 判定與輸出
            # 如果相似度不足且還有組件，嘗試進一步拆解
            if (max_sim < accept_sim or second_sim < second_accept_sim) and depth < max_depth and len(cluster.components) > 1:
                rec_out.extend(recurse(tgt, depth + 1, topleft=next_topleft))
            else:
                if max_sim < accept_sim and best_macro is not None:
                    print(f"Doubted: {best_macro.macro} (Sim: {max_sim:.3f})")
                rec_out.append(
                    AdaptiveReturnSet(cluster, best_macro, max_sim, second_sim, topleft, depth)
                )
                
        return rec_out
    
    return recurse(img)

def test(case: str):
    filein = f"./testcases/case{case}.png"
    fileout = f"./out/case{case}.png"
    
    sauce = load_tex_macros("./templates")

    src_img = Image.open(filein)
    src_img = src_img.convert("L")
    enhancer = ImageEnhance.Contrast(src_img)
    src_img = enhancer.enhance(2.0)

    import time 
    start = time.time()
    out = adaptive_cluster(
        src_img, sauce, hausdorff_similarity, second_accept_sim = 0.85
    )
    print(time.time() - start)

    src_img = src_img.convert("RGB")
    myTable = PrettyTable(
        ["Position (y, x, h, w)", "Centroid (x, y)", 
         "Answer", "Similarity", "2nd Similarity", "Depth"]
    )
    tableRows = []
    for result in out:
        dy, dx = result.topleft
        y, x, h, w = result.cluster.get_bbox_yxhw()
        cx, cy = result.cluster.get_centroid()
        tableRows.append([
            (y+dy, x+dx, h, w), 
            (cx+dx, cy+dy), 
            result.best_macro.macro, 
            result.max_sim, 
            result.second_sim, 
            result.depth
        ])
        draw = ImageDraw.Draw(src_img, "RGB")
        draw.rectangle((x+dx-1, y+dy-1, x+dx+w+1, y+dy+h+1), None, (0,0,128) if result.max_sim >= 0.7 else (128,0,0))
    tableRows.sort(
        key = lambda row: row[1][0]
    )
    for row in tableRows:
        myTable.add_row(row)
    src_img.save(fileout)
    print(myTable)

if __name__ == "__main__":
    test("01")

