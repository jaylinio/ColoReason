import json
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
from openpyxl import Workbook
from collections import defaultdict


def calculate_similarity_avg(input_file):
    """Run calculate similarity avg."""

    font_path = ""
    if not font_path or not os.path.exists(font_path):
        raise FileNotFoundError("font_path must reference an existing font file")

    my_font = font_manager.FontProperties(fname=font_path)

    target_fields = ["ICD预测", "再入院概率", "药物推荐"]
    field_values = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line.strip())
            similarity = data.get("similarity", {})

            if not isinstance(similarity, dict):
                continue

            for field in target_fields:
                val = similarity.get(field)
                if isinstance(val, (int, float)):
                    field_values[field].append(val)

    field_averages = {
        f: (sum(vals) / len(vals)) if vals else 0.0 for f, vals in field_values.items()
    }
    field_counts = {f: len(vals) for f, vals in field_values.items()}

    print("=== Average similarity ===")
    for k, v in field_averages.items():
        print(f"{k}: {v:.2f}")

    print("=== Valid numeric value count ===")
    for k, c in field_counts.items():
        print(f"{k}: {c}")

    plt.figure(figsize=(7, 6))
    fields = list(field_averages.keys())
    averages = list(field_averages.values())

    bars = plt.bar(fields, averages, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(0, 100)
    plt.ylabel("Average similarity", fontproperties=my_font)
    plt.title("Average similarity across three tasks", fontproperties=my_font)
    plt.xticks(fontproperties=my_font)
    plt.yticks(fontproperties=my_font)
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontproperties=my_font,
            fontsize=12,
        )

    png_path = input_file.replace(".jsonl", ".png")
    xlsx_path = input_file.replace(".jsonl", ".xlsx")

    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved bar chart to: {png_path}")

    wb = Workbook()
    ws = wb.active
    ws.title = "similarity"
    ws.append(["Field", "Average similarity"])

    for k in target_fields:
        avg = round(field_averages.get(k, 0.0), 2)
        ws.append([k, avg])

    wb.save(xlsx_path)
    print(f"Saved score table to: {xlsx_path}")


if __name__ == "__main__":
    input_file = ""
    calculate_similarity_avg(input_file)
