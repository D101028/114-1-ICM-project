import time
from typing import List, Callable

from PIL import Image, ImageEnhance, ImageDraw
from prettytable import PrettyTable

from src.cluster import ClusterGroup, extract_components_from_pil, hierarchical_cluster
from src.chamfer import chamfer_similarity
from src.hausdorff import hausdorff_similarity
from src.l2dist import l2_similarity
from src.template import TexMacro, load_tex_macros

class AdaptiveReturnSet:
    def __init__(self, cluster, best_macro, max_sim, second_sim, depth) -> None:
        self.cluster = cluster
        self.best_macro = best_macro
        self.max_sim = max_sim
        self.second_sim = second_sim
        self.depth = depth

def adaptive_cluster(
    input_img: Image.Image, sauce: List[TexMacro], 
    sim_func: Callable[[ClusterGroup, ClusterGroup], float] = l2_similarity, 
    second_sim_func: Callable[[ClusterGroup, ClusterGroup], float] = lambda _, __: 1.0, 
    accept_sim = 0.7, second_accept_sim = 0.9, max_depth = 16
) -> List[AdaptiveReturnSet]:

    def recurse(cluster: ClusterGroup, depth = 0, fix_ratio_x = 1.0, fix_ratio_y = 1.0) -> List[AdaptiveReturnSet]:
        # 1. 初始化參數與分群
        wx, wy = 2 + 0.1 * fix_ratio_x, 0.1 + 0.1 * fix_ratio_y
        
        clusters = hierarchical_cluster(
            cluster,
            base_threshold=cluster.get_bbox_hw()[0] * 0.4,
            wx=wx, wy=wy,
            refine_min_size=4, refine_ratio=0.5
        )

        # 2. 特殊情況：如果分不開且還有深度，增加權重重試
        if len(clusters) == 1 and len(clusters[0].components) > 1 and depth < max_depth:
            return recurse(clusters[0], depth + 1, fix_ratio_x + 0.5, fix_ratio_y + 2)
        
        rec_out: List[AdaptiveReturnSet] = []
        for cluster in clusters:
            h, w = cluster.get_bbox_hw()
            r = h / w

            # 3. 遞迴深挖：如果組件過多，強制進入下一層
            if len(cluster.components) > 4 and depth < max_depth:
                rec_out.extend(recurse(cluster, depth + 1))
                continue

            # 4. 統一比對邏輯
            best_macro, max_sim, second_sim = None, 0.0, 0.0
            
            for tex_macro in sauce:
                # 比率篩選邏輯
                is_compatible = (
                    (r < 0.1 and tex_macro.yx_ratio <= 0.125) or
                    (r > 10 and tex_macro.yx_ratio >= 8) or
                    (0.1 <= r <= 10 and 0.1 <= tex_macro.yx_ratio <= 10)
                )
                if not is_compatible: continue

                l2_sim = sim_func(tex_macro.cluster, cluster)
                
                if l2_sim > max_sim:
                    max_sim = l2_sim
                    second_sim = second_sim_func(tex_macro.cluster, cluster)
                    best_macro = tex_macro

            # 5. 判定與輸出
            # 如果相似度不足且還有組件，嘗試進一步拆解
            if (max_sim < accept_sim or second_sim < second_accept_sim) and depth < max_depth and len(cluster.components) > 1:
                rec_out.extend(recurse(cluster, depth + 1))
            else:
                if max_sim < accept_sim and best_macro is not None:
                    print(f"Doubted: {best_macro.macro} (Sim: {max_sim:.3f})")
                rec_out.append(
                    AdaptiveReturnSet(cluster, best_macro, max_sim, second_sim, depth)
                )
                
        return rec_out
    
    components, _, _ = extract_components_from_pil(input_img)
    cluster = ClusterGroup(components)

    return recurse(cluster)

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(4.0)
    return img

def test(case: str):
    filein = f"./testcases/case{case}.png"
    fileout = f"./out/case{case}.png"
    
    sauce = load_tex_macros("./templates")

    src_img = load_image(filein)

    start = time.time()
    out = adaptive_cluster(
        src_img, sauce, 
        sim_func=l2_similarity, 
        second_sim_func=hausdorff_similarity, 
        accept_sim=0.6, 
        second_accept_sim=0.9
    )
    print(time.time() - start)

    src_img = src_img.convert("RGB")
    myTable = PrettyTable(
        ["Position (y, x, h, w)", "Centroid (x, y)", 
         "Answer", "Similarity", "2nd Similarity", "Depth"]
    )
    tableRows = []
    for result in out:
        y, x = result.cluster.topleft
        h, w = result.cluster.get_bbox_hw()
        cx, cy = result.cluster.get_centroid()
        tableRows.append([
            (y, x, h, w), 
            (cx, cy), 
            result.best_macro.macro, 
            result.max_sim, 
            result.second_sim, 
            result.depth
        ])
        draw = ImageDraw.Draw(src_img, "RGB")
        draw.rectangle((x-1, y-1, x+w+1, y+h+1), None, (0,0,128) if result.max_sim >= 0.7 else (128,0,0))
        draw.circle((cx, cy), 1, (0,255,0))

    tableRows.sort(
        key = lambda row: row[1][0]
    )
    for row in tableRows:
        myTable.add_row(row)
    src_img.save(fileout)
    print(myTable)

def test1(case: str, N: int = 10):
    """純 l2 sim 實驗"""
    sauce = load_tex_macros("templates")
    src_img = load_image(f"testcases/case{case}.png")

    start = time.time()
    for _ in range(N):
        adaptive_cluster(
            src_img, sauce
        )
    end = time.time()

    print(f"Time in Average: {(end - start) / N:.4f}s")

if __name__ == "__main__":
    test("05")

