from scipy.stats import wilcoxon


accuracy_gp  = [0.7094, 0.9582,0.7294, 0.9782, 0.6914, 0.8612]
accuracy_mlp = [0.6533, 0.4753,0.6033, 0.5053, 0.5733, 0.5142]

fscore_gp = [0.6239, 0.3551, 0.6014, 0.3312, 0.6701, 0.4010]
fscore_mlp  = [0.6759, 0.9581,0.6169, 0.9010,0.6099, 0.9011]

acc_stat, acc_p = wilcoxon(accuracy_gp, accuracy_mlp)
print(f"Accuracy Wilcoxon: stat={acc_stat}, p={acc_p}")


fscore_stat, fscore_p = wilcoxon(fscore_gp, fscore_mlp)
print(f"F-Score Wilcoxon: stat={fscore_stat}, p={fscore_p}")

