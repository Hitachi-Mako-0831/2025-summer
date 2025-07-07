import json
import os
from binoculars import Binoculars
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix


bino = Binoculars()

def load_data(file_path="code_data.json"):
    texts = []
    is_gpt_source = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Ensure the loaded data is a list of dictionaries
            if not isinstance(data, list):
                # If it's a single dictionary, wrap it in a list for processing
                data = [data]

            for item in data:
                texts.append(item["Text"])
                is_gpt_source.append(item["Source"] == "GPT")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return texts, is_gpt_source

def fix_result(results):
    return [result == "Most likely AI-generated" for result in results]

if __name__ == "__main__":
    file_path = "./dataset/Yelp Review/yelp_combine.json"
    texts, is_gpt_source = load_data(file_path)
    BATCH_SIZE = 20 # 或者 2, 4, ... 尝试一个很小的数值
    all_results = []
    print(f"\nProcessing {len(texts)} texts in batches of {BATCH_SIZE}...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_texts)} items)...")
        # Ensure that the model is indeed on GPU and not accidentally moved to CPU if doing manual batching
        batch_predictions = bino.predict(batch_texts)
        all_results.extend(batch_predictions)
        # Consider adding torch.cuda.empty_cache() here, though it's often done internally
        # torch.cuda.empty_cache() 
    results = all_results
    print("Batch processing complete.")
    fixed_result = fix_result(results)
    
    if len(is_gpt_source) != len(fixed_result):
        print("Error: Length of true labels and predictions do not match. Cannot calculate metrics.")
    else:
            # Calculate metrics
            # For these metrics, it's crucial to define what is "positive" (P) and "negative" (N)
            # Let's define "GPT-generated" as the positive class (True).

            # Accuracy
        accuracy = accuracy_score(is_gpt_source, fixed_result)

            # Precision, Recall (TPR), F1-Score
            # `pos_label=True` means 'True' in is_gpt_source and fixed_result is considered the positive class (GPT-generated)
            # `zero_division=0` handles cases where precision/recall might be undefined (e.g., no true positives)
        precision = precision_score(is_gpt_source, fixed_result, pos_label=True, zero_division=0)
        recall = recall_score(is_gpt_source, fixed_result, pos_label=True, zero_division=0) # TPR
        f1 = f1_score(is_gpt_source, fixed_result, pos_label=True, zero_division=0)

            # Confusion Matrix to calculate FPR
            # CM layout: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(is_gpt_source, fixed_result, labels=[False, True]).ravel()
            
            # False Positive Rate (FPR) = FP / (FP + TN)
            # Denominator (FP + TN) is the total number of actual negatives (Human-written)
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0


        print("\n--- Model Evaluation Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (for 'GPT-generated' class): {precision:.4f}")
        print(f"Recall (TPR - for 'GPT-generated' class): {recall:.4f}")
        print(f"F1 Score (for 'GPT-generated' class): {f1:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
            
        print("\n--- Confusion Matrix ---")
        print(f"True Negatives (TN): {tn}") # Human correctly predicted as Human
        print(f"False Positives (FP): {fp}") # Human incorrectly predicted as GPT
        print(f"False Negatives (FN): {fn}") # GPT incorrectly predicted as Human
        print(f"True Positives (TP): {tp}") # GPT correctly predicted as GPT

        output_data = {
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall_tpr": recall,
                "f1_score": f1,
                "fpr": fpr,
                "confusion_matrix": {
                    "tn": int(tn), # Convert to int as np.int64 is not JSON serializable
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                }
            },
            "individual_results": []
        }

        for i in range(len(texts)):
            output_data["individual_results"].append({
                "text": texts[i],
                "true_label_is_gpt": is_gpt_source[i],
                "raw_prediction": results[i],
                "fixed_prediction_is_gpt": fixed_result[i]
            })
            
        output_file_name = "test_results_yelp.json"
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"\nTest results saved to '{output_file_name}'")
        except IOError as e:
            print(f"Error saving test results to '{output_file_name}': {e}")
        # --- End Save test results ---
