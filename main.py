import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

# === 讀取 Excel 資料 ===
FILE_PATH = "食品營養成分資料庫2024UPDATE2.xlsx"

columns_needed = [
    "食品分類","樣品名稱","俗名","熱量(kcal)", "修正熱量(kcal)", "水分(g)", "粗蛋白(g)", "粗脂肪(g)",
    "飽和脂肪(g)", "灰分(g)", "總碳水化合物(g)", "膳食纖維(g)"
]

df = pd.read_excel(FILE_PATH, usecols=columns_needed)

# 去除缺值
df = df.dropna(subset=["熱量(kcal)", "粗蛋白(g)", "粗脂肪(g)", "總碳水化合物(g)"])

# 建立 FOOD_DATA 字典
FOOD_DATA = {}
for _, row in df.iterrows():
    name = str(row["俗名"]).strip() if not pd.isna(row["俗名"]) else str(row["樣品名稱"]).strip()
    FOOD_DATA[name] = {
        "calories": float(row["熱量(kcal)"]),
        "protein": float(row["粗蛋白(g)"]),
        "carbs": float(row["總碳水化合物(g)"]),
        "fat": float(row["粗脂肪(g)"]),
    }

print(f"✅ 已載入 {len(FOOD_DATA)} 種食物營養資料。")

# === 匈牙利演算法找比例最相似的前五個食物 ===
def find_top5_similar_foods(user_input, food_data):
    """
    以比例（相對營養比）比較相似度，輸出前五個最相近的食物
    """
    # 將使用者輸入轉為 numpy 向量
    user_vec = np.array([
        user_input["calories"],
        user_input["protein"],
        user_input["carbs"],
        user_input["fat"]
    ])
    # 避免除以零
    user_vec[user_vec == 0] = 1e-6

    # 將使用者資料轉為比例向量
    user_ratio = user_vec / np.sum(user_vec)

    food_names = list(food_data.keys())
    food_matrix = np.array([
        [f["calories"], f["protein"], f["carbs"], f["fat"]] for f in food_data.values()
    ])

    # 每個食物也轉為比例
    food_ratios = food_matrix / food_matrix.sum(axis=1, keepdims=True)

    # 計算距離矩陣 (比例差距)
    cost_matrix = np.abs(food_ratios - user_ratio)

    # 匈牙利演算法：這裡可用於平衡匹配，但我們僅需計算總距離排序
    total_cost = cost_matrix.sum(axis=1)

    # 找出前五名最小距離的食物
    top5_indices = np.argsort(total_cost)[:5]
    top5_results = [(food_names[i], total_cost[i], food_data[food_names[i]]) for i in top5_indices]

    return top5_results


# === 測試輸入（可換成你自己的數值）===
user_input = {
    "calories": 365,
    "protein": 8.6,
    "carbs": 77.1,
    "fat": 1.6
}

top5 = find_top5_similar_foods(user_input, FOOD_DATA)

print("\n🍎 與你輸入的營養比例最相近的前五種水果：\n")
for rank, (name, diff, info) in enumerate(top5, start=1):
    print(f"{rank}. {name}  (距離值: {diff:.4f})")
    print(f"   → 熱量 {info['calories']} kcal, 蛋白質 {info['protein']} g, 碳水 {info['carbs']} g, 脂肪 {info['fat']} g\n")
