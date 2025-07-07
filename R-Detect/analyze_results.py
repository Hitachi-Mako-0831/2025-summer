import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def analyze_evaluation_results(json_file_path):
    """
    从 evaluation_results.json 文件中读取数据，并重新计算各项评估指标。

    Args:
        json_file_path (str): evaluation_results.json 文件的路径。
    """
    print(f"Loading results from: {json_file_path}")

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            all_results_detail = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Make sure it's valid JSON.")
        return

    true_labels = []
    predicted_labels = []

    # 定义标签映射，与 run_evaluation.py 保持一致
    LABEL_MAP = {
        "human": 0,    # 负类
        "machine": 1,  # 正类
    }

    PREDICTION_MAP = {
        "Most likely Human Write": 0,
        "Most likely Machine Generated": 1,
    }

    # 遍历加载的数据，提取真实标签和预测标签
    for entry in all_results_detail:
        # 跳过有错误条目的数据
        if "error" in entry:
            # print(f"Skipping entry due to error: {entry.get('filename', 'Unknown File')}")
            continue

        true_label_str = entry.get("true_label_str")
        predicted_result_str = entry.get("predicted_result_str")

        if true_label_str is None or predicted_result_str is None:
            print(f"Warning: Missing true_label_str or predicted_result_str in entry: {entry.get('filename', 'Unknown File')}. Skipping.")
            continue

        # 映射到数值
        current_true_label_int = LABEL_MAP.get(true_label_str)
        
        current_predicted_label_int = None
        if "Human Write" in predicted_result_str:
            current_predicted_label_int = PREDICTION_MAP["Most likely Human Write"]
        elif "Machine Generated" in predicted_result_str:
            current_predicted_label_int = PREDICTION_MAP["Most likely Machine Generated"]
        
        if current_true_label_int is None or current_predicted_label_int is None:
            print(f"Warning: Could not map labels for {entry.get('filename', 'Unknown File')}. Skipping.")
            continue

        true_labels.append(current_true_label_int)
        predicted_labels.append(current_predicted_label_int)

    print(f"Successfully processed {len(true_labels)} entries for metric calculation.")

    if len(true_labels) == 0:
        print("No valid predictions were found after filtering. Cannot calculate metrics.")
        return

    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)

    print("\n--- Recalculated Evaluation Summary ---")

    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # 精确率 (Precision)
    precision_machine = precision_score(y_true, y_pred, pos_label=1, average='binary')
    print(f"Precision (Machine Generated): {precision_machine:.4f}")
    precision_human = precision_score(y_true, y_pred, pos_label=0, average='binary')
    print(f"Precision (Human Write): {precision_human:.4f}")

    # 召回率 (Recall)
    recall_machine = recall_score(y_true, y_pred, pos_label=1, average='binary')
    print(f"Recall (Machine Generated): {recall_machine:.4f}")
    recall_human = recall_score(y_true, y_pred, pos_label=0, average='binary')
    print(f"Recall (Human Write): {recall_human:.4f}")

    # F1-score
    f1_machine = f1_score(y_true, y_pred, pos_label=1, average='binary')
    print(f"F1-score (Machine Generated): {f1_machine:.4f}")
    f1_human = f1_score(y_true, y_pred, pos_label=0, average='binary')
    print(f"F1-score (Human Write): {f1_human:.4f}")

    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"F1-score (Macro Avg): {f1_macro:.4f}")

    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-score (Weighted Avg): {f1_weighted:.4f}")

    # 计算混淆矩阵并提取 TPR 和 FPR
    # 这里的 pos_label=1 (Machine Generated), neg_label=0 (Human Write)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
    print(f"True Positive Rate (TPR / Recall MGT): {tpr:.4f}")

    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
    print(f"False Positive Rate (FPR): {fpr:.4f}")


if __name__ == "__main__":
    # 假设 evaluation_results.json 在当前目录下
    results_file = "evaluation_results_code.json"
    analyze_evaluation_results(results_file)

    # 如果你的结果文件在其他地方，比如 "path/to/your/evaluation_results.json"
    # results_file_alt = "./some_other_folder/evaluation_results.json"
    # analyze_evaluation_results(results_file_alt)