import matplotlib.pyplot as plt


N = [1, 2, 3, 4, 5]
separate = [203, 240, 259, 270, 284]      
concatenated = [169, 214, 240, 260, 269]  


plt.figure(figsize=(12, 8))
plt.plot(N, separate, marker='o', color='blue', label='Separate Embedding')
plt.plot(N, concatenated, marker='o', color='green', label='Concatenated Embedding')


# plt.title("FL performance comparison")
plt.xlabel("N", fontsize=16)
plt.ylabel("Top-N", fontsize=16)


plt.xticks(N)


plt.legend(fontsize=15)


plt.grid(True, linestyle='--', alpha=0.6)


plt.savefig("../results/figures/ablation2.png", dpi=300, bbox_inches='tight')

