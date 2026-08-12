import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from matplotlib import font_manager
from openpyxl import Workbook


def calculate_average_and_plot(input_file):

    font_path = ""
    if not font_path or not os.path.exists(font_path):
        raise FileNotFoundError("font_path must reference an existing font file")

    my_font = font_manager.FontProperties(fname=font_path)

    field_values = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            similarity = data.get("similarity", {})
            metadata = data.get("metadata", {})
            field = metadata.get("field", "")

            if field and isinstance(similarity, dict):
                value = similarity.get(field, None)
                if value is not None:
                    if isinstance(value, (int, float)):
                        field_values[field].append(value)

    field_averages = {
        f: (sum(vals) / len(vals)) for f, vals in field_values.items() if vals
    }

    items = field_averages.items()
    fields = [k for k, _ in items]
    averages = [v for _, v in items]

    print("Average score by field:")
    for k, v in items:
        print(f"{k}: {v:.2f}")

    plt.figure(figsize=(16, 8))
    plt.bar(fields, averages)
    plt.xticks(rotation=45, ha="right", fontproperties=my_font)
    plt.yticks(fontproperties=my_font)
    plt.xlabel("Field", fontproperties=my_font)
    plt.ylabel("Average similarity", fontproperties=my_font)
    plt.title("Average similarity by field", fontproperties=my_font)
    plt.tight_layout()

    png_path = input_file.replace(".jsonl", ".png")
    xlsx_path = input_file.replace(".jsonl", ".xlsx")

    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved bar chart to: {png_path}")

    wb = Workbook()
    ws = wb.active
    ws.title = "scores"
    ws.append(["Field", "Score"])
    for k, v in items:
        ws.append([k, round(v, 2)])
    wb.save(xlsx_path)
    print(f"Saved score table to: {xlsx_path}")


if __name__ == "__main__":
    input_file = ""
    calculate_average_and_plot(input_file)
