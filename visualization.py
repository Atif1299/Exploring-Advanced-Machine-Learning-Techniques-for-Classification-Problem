import math
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_columns(data, selected_features):
    num_cols = len(selected_features)
    max_cols_per_row = 6
    n_rows = math.ceil(num_cols / max_cols_per_row)

    fig, axes = plt.subplots(n_rows, max_cols_per_row, figsize=(max_cols_per_row * 3, n_rows * 2.5))
    axes = axes.flatten()

    for i, col in enumerate(selected_features):
        if data[col].dtype in ["int64", "float64"]:
            sns.histplot(data[col], kde=True, ax=axes[i], color="blue")
            axes[i].set_title(f"Distribution of {col}")
        else:
            sns.countplot(data=data, y=col, ax=axes[i], color="blue")
            axes[i].set_title(f"Countplot of {col}")

    for i in range(num_cols, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2.0)
    plt.show()
