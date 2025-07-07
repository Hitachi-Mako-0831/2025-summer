# run_evaluation.py
import os
import subprocess
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np

# --- 配置路径和参数 ---
DATASET_ROOT_DIR = "./my_code_dataset/" # 你的数据集根目录
TEST_DIR = os.path.join(DATASET_ROOT_DIR, "test") # 测试集目录
HWT_REF_PATH = "./my_feature_refs/feature_ref_HWT_code_data.pt" # 新生成的 HWT 特征参考
MGT_REF_PATH = "./my_feature_refs/feature_ref_MGT_code_data.pt" # 新生成的 MGT 特征参考
LOCAL_MODEL_PATH = "./llm-models/roberta-base"


# 定义标签映射
# 注意：你的模型输出是 "Most likely Human Write" 和 "Most likely Machine Generated"
# 我们需要将它们映射到 0 和 1
LABEL_MAP = {
    "human": 0,    # 真实标签 Human 映射为 0
    "machine": 1,  # 真实标签 Machine 映射为 1
}

PREDICTION_MAP = {
    "Most likely Human Write": 0,
    "Most likely Machine Generated": 1,
}

# --- 存储结果 ---
true_labels = []
predicted_labels = []
all_results_detail = []

print(f"Starting evaluation on test set in: {TEST_DIR}")
print(f"Using HWT ref: {HWT_REF_PATH}")
print(f"Using MGT ref: {MGT_REF_PATH}")
print(f"Using local model: {LOCAL_MODEL_PATH}")

# 遍历测试目录中的每个文件
for filename in os.listdir(TEST_DIR):
    if filename.endswith(".txt"):
        test_file_path = os.path.join(TEST_DIR, filename)
        
        # --- 推断真实标签 ---
        # 假设文件名包含 "human" 或 "machine" 来指示真实标签
        current_true_label_str = ""
        if "human" in filename.lower():
            current_true_label_str = "human"
        elif "machine" in filename.lower():
            current_true_label_str = "machine"
        else:
            print(f"Warning: Could not infer true label for {filename}. Skipping.")
            continue # 如果无法推断标签，则跳过此文件

        current_true_label_int = LABEL_MAP[current_true_label_str]
        
        print(f"\n--- Testing: {filename} (True Label: {current_true_label_str}) ---")
        
        # --- 构建 main.py 的命令行命令 ---
        command = [
            "python", "./main.py",
            "--test_file", test_file_path,
            "--use_gpu", # 如果你希望强制使用 GPU
            "--local_model", LOCAL_MODEL_PATH,
            "--feature_ref_HWT", HWT_REF_PATH,
            "--feature_ref_MGT", MGT_REF_PATH,
            # "--threshold", str(DEFAULT_THRESHOLD) # 如果 main.py 支持该参数，请取消注释
        ]

        # --- 执行命令并捕获输出 ---
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            output_lines = process.stdout.strip().split('\n')
            
            # 提取预测结果，通常是最后一行
            predicted_result_str = output_lines[-1].strip() 
            print(f"Raw Prediction Output: {predicted_result_str}")

            # --- 映射预测结果到数值标签 ---
            current_predicted_label_int = None
            if "Human Write" in predicted_result_str:
                current_predicted_label_int = PREDICTION_MAP["Most likely Human Write"]
            elif "Machine Generated" in predicted_result_str:
                current_predicted_label_int = PREDICTION_MAP["Most likely Machine Generated"]
            else:
                print(f"Warning: Unrecognized prediction format for {filename}: {predicted_result_str}. Skipping.")
                continue

            print(f"Mapped Predicted Label: {'Human' if current_predicted_label_int == 0 else 'Machine'}")

            true_labels.append(current_true_label_int)
            predicted_labels.append(current_predicted_label_int)
            
            all_results_detail.append({
                "filename": filename,
                "true_label_str": current_true_label_str,
                "true_label_int": current_true_label_int,
                "predicted_result_str": predicted_result_str,
                "predicted_label_int": current_predicted_label_int,
                "full_output": process.stdout # 保存完整输出，方便调试
            })

        except subprocess.CalledProcessError as e:
            print(f"Error running main.py for {filename}:")
            print(e.stderr)
            all_results_detail.append({
                "filename": filename,
                "error": str(e),
                "stderr": e.stderr,
                "stdout": e.stdout
            })
        except Exception as e:
            print(f"An unexpected error occurred for {filename}: {e}")
            all_results_detail.append({
                "filename": filename,
                "error": str(e)
            })

# --- 4. 计算评估指标 ---
print("\n--- Evaluation Summary ---")

if len(true_labels) == 0:
    print("No valid predictions were made. Cannot calculate metrics.")
else:
    # 将列表转换为 NumPy 数组以便计算
    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)

    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # 精确率 (Precision)
    # average='binary' 适用于二分类，pos_label=1 表示我们关注的“正类”是机器生成 (MGT)
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
    # average='binary' 同样，pos_label=1 表示我们关注 MGT 的 F1-score
    f1_machine = f1_score(y_true, y_pred, pos_label=1, average='binary')
    print(f"F1-score (Machine Generated): {f1_machine:.4f}")
    f1_human = f1_score(y_true, y_pred, pos_label=0, average='binary')
    print(f"F1-score (Human Write): {f1_human:.4f}")

    # 宏平均 F1-score (对每个类别的 F1-score 进行简单平均，不考虑样本数)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"F1-score (Macro Avg): {f1_macro:.4f}")

    # 加权平均 F1-score (考虑每个类别的样本数进行加权平均)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-score (Weighted Avg): {f1_weighted:.4f}")

    # --- 计算混淆矩阵并提取 TPR 和 FPR ---
    # conf_matrix 的顺序是 [[TN, FP], [FN, TP]]
    # 这里的 pos_label=1 (Machine Generated), neg_label=0 (Human Write)
    # y_true (真实标签) vs y_pred (预测标签)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # 确保标签顺序是 [负类, 正类]
    
    TN, FP, FN, TP = cm.ravel() # 将矩阵展平为 TN, FP, FN, TP

    # 计算 TPR (True Positive Rate) / Recall (Machine Generated)
    # TP / (TP + FN)
    tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
    print(f"True Positive Rate (TPR / Recall MGT): {tpr:.4f}")

    # 计算 FPR (False Positive Rate)
    # FP / (FP + TN)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    # 可以将详细结果保存到文件
    output_results_file = "evaluation_results_code.json"
    with open(output_results_file, "w", encoding='utf-8') as f:
        json.dump(all_results_detail, f, indent=4)
    print(f"\nDetailed results for each file saved to {output_results_file}")