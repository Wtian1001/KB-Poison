import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_perplexity(filename):
    perplexities = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            perplexities.append(obj["perplexity"])
    return perplexities


normal_perplexities = load_perplexity("ms_clean_ppl.json")
poisoned_perplexities = load_perplexity("ms_ppl.json")


plt.figure(figsize=(8, 5))
plt.hist(normal_perplexities, bins=30, alpha=0.5, label='Normal', density=True)
plt.hist(poisoned_perplexities, bins=30, alpha=0.5, label='Poisoned', density=True)
plt.xlabel('Perplexity')
plt.ylabel('Density')
plt.legend()
plt.title('Perplexity Distribution Histogram')
plt.show()


plt.figure(figsize=(8, 5))
sns.kdeplot(normal_perplexities, label='Normal', shade=True)
sns.kdeplot(poisoned_perplexities, label='Poisoned', shade=True)
plt.xlabel('Perplexity')
plt.ylabel('Density')
plt.title('Perplexity KDE Distribution')
plt.legend()
plt.show()


plt.figure(figsize=(6, 5))
plt.boxplot([normal_perplexities, poisoned_perplexities], labels=['Normal', 'Poisoned'])
plt.ylabel('Perplexity')
plt.title('Boxplot of Perplexity')
plt.show()