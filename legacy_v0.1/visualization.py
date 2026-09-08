import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os

plt.rcParams['axes.axisbelow'] = True
CUSTOM_COLORS = ['#ff9999', '#66b3ff', '#99ff99']

def graph_distribution(df, output_path="results/plots/yeardistribution_processed.png"):
    plt.figure(figsize=(6,4))
    plt.hist(df["Publication Year"], bins=8)
    plt.grid(alpha=0.5)
    plt.xlabel("Year")
    plt.ylabel("Paper count")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130)
    plt.close()

def graph_sources(df, output_path="results/plots/barplot_processed.png"):
    sources_df = df.groupby("Source").count()["Title"]
    plt.figure(figsize=(12,4))
    ax = sources_df.plot(kind="barh")
    ax.set_xlabel("Paper counts")
    plt.grid(alpha=0.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_class_distributions_by_model(df, models, labels_map, output_dir="results/plots"):
    """
    labels_map: dict of {abbreviation: full_label_string}
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = list(labels_map.keys())

    for model in models:
        # Create boolean flags in bulk to avoid fragmentation
        new_cols = {}
        for abbr in labels:
            # Score columns are named {abbr}_{model} in classified.csv
            score_col = f"{abbr}_{model}"
            if score_col in df.columns:
                new_cols[f"{abbr}_{model}_bool"] = df[score_col].astype(float) > 0.875
        
        # Add all boolean columns at once
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        trues = []
        for abbr in labels:
            bool_col = f"{abbr}_{model}_bool"
            if bool_col in df.columns:
                true_count = df[bool_col].sum()
                trues.append(100 * true_count / len(df))
            else:
                trues.append(0.0)

        z = [x for _, x in sorted(zip(trues, labels), reverse=True)]
        sorted_trues = sorted(trues, reverse=True)

        plt.figure(figsize=(16,4))
        plt.bar(z, sorted_trues, color=CUSTOM_COLORS[1])
        for i in range(len(labels)):
            plt.text(z[i], sorted_trues[i]/2, "{:02.1f}".format(sorted_trues[i]), ha="center")
        plt.grid()
        plt.xlabel("Topic Abbreviation")
        plt.ylabel("Percentage of papers")
        plt.text(0.9, 0.9, "Total papers: "+str(len(df)), size=10, transform=plt.gca().transAxes,
                ha="center", va="center",
                bbox=dict(boxstyle="round", ec=(0., 0.5, 0.5), fc=(1., 1, 1)))
        
        plt.savefig(os.path.join(output_dir, f"overall_{model}.png"))
        plt.close()

    return df

def get_overall_classes(df, models, labels_map, output_path="results/plots/overall.png"):
    labels = list(labels_map.keys())
    trues = []
    
    new_overall_cols = {}
    for abbr in labels:
        # Check if any model classified this label as true
        bool_cols = [f"{abbr}_{model}_bool" for model in models if f"{abbr}_{model}_bool" in df.columns]
        if bool_cols:
            new_overall_cols[abbr] = df[bool_cols].any(axis=1)
        else:
            new_overall_cols[abbr] = pd.Series([False] * len(df), index=df.index)
        
        true_count = new_overall_cols[abbr].sum()
        trues.append(100 * true_count / len(df))

    # Add overall selection columns at once
    df = pd.concat([df, pd.DataFrame(new_overall_cols, index=df.index)], axis=1)

    z = [x for _, x in sorted(zip(trues, labels), reverse=True)]
    sorted_trues = sorted(trues, reverse=True)

    plt.figure(figsize=(16,4))
    plt.bar(z, sorted_trues, color=CUSTOM_COLORS[1])
    for i in range(len(labels)):
        plt.text(z[i], sorted_trues[i]/2, "{:02.1f}".format(sorted_trues[i]), ha="center")
    plt.grid()
    plt.xlabel("Topic Abbreviation")
    plt.ylabel("Percentage of papers")
    plt.text(0.9, 0.9, "Total papers: "+str(len(df)), size=10, transform=plt.gca().transAxes,
            ha="center", va="center",
            bbox=dict(boxstyle="round", ec=(0., 0.5, 0.5), fc=(1., 1, 1)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    return df
