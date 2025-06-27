import json

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def load_perplexity(filename):
    with open(filename, 'r') as f:
        return [json.loads(line.strip())["perplexity"] for line in f]


normal_perplexities = load_perplexity("ms_clean_ppl.json")
poisoned_perplexities = load_perplexity("ms_ppl_1.json")


y_true = [0] * len(normal_perplexities) + [1] * len(poisoned_perplexities)  # 0=正常, 1=poisoned
scores = normal_perplexities + poisoned_perplexities  # 用perplexity做判别分数


fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (perplexity-based detection) for MSMACRO Dataset")
plt.legend(loc="lower right")
plt.show()

