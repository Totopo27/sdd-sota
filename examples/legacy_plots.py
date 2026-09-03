import pandas as pd
import numpy as np
import os
import argparse
from litreview.visualization import (
    graph_distribution, 
    graph_sources, 
    plot_class_distributions_by_model, 
    get_overall_classes
)
from litreview.utils import load_config

def filter_and_save_subsets(df, processed_dir, labels_map):
    """Specific filtering logic updated to use labels from config."""
    # We use abbreviated keys from labels_map if possible, or assume defaults
    nmd_key = "NMD"
    hpc_key = "HPC"
    stc_key = "STC"
    dim2_key = "2D"

    if nmd_key not in df.columns:
        print(f"{nmd_key} column missing. Make sure classification happened.")
        return
        
    df_NMD = df[df[nmd_key]]
    # we only want the ones that have to dow with 2 dimensional modeling
    df_2D = df_NMD[df_NMD[dim2_key]]
    # We only consider the ones that are not explicitly stochastic
    df_2D = df_2D[np.logical_not(df_2D[stc_key])]
    
    print(f"HPC distribution in 2D NMD deterministic papers:\n{np.unique(df_2D[hpc_key], return_counts=True)}")
    
    os.makedirs(processed_dir, exist_ok=True)
    df_2D[["Title", "Abstract Note"]].to_csv(os.path.join(processed_dir, "2D_read.csv"), index=False)
    df_2D[np.logical_not(df_2D[hpc_key])][["Title", "Abstract Note"]].to_csv(os.path.join(processed_dir, "notHPC.csv"), index=False)
    df_2D[df_2D[hpc_key]][["Title", "Abstract Note"]].to_csv(os.path.join(processed_dir, "2DHPC.csv"), index=False)
    print(f"Filtered CSVs saved to {processed_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate plots for literature review.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    processed_dir = paths.get("processed_data", "data/processed")
    plots_dir = paths.get("plots", "results/plots")
    
    input_file = os.path.join(processed_dir, "classified.csv")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Run scripts/run_analysis.py first.")
        return

    df = pd.read_csv(input_file)
    
    # Extract model names from config for plotting
    model_names = [m.split("/")[-1] for m in config.get("models", [])]
    labels_map = config.get("labels", {})

    print("Generating source distribution plot...")
    graph_sources(df, output_path=os.path.join(plots_dir, "barplot_processed.png"))
    
    print("Generating year distribution plot...")
    graph_distribution(df, output_path=os.path.join(plots_dir, "yeardistribution_processed.png"))

    print("Generating model-specific topic distributions...")
    df = plot_class_distributions_by_model(df, model_names, labels_map, output_dir=plots_dir)

    print("Generating overall topic distribution...")
    df = get_overall_classes(df, model_names, labels_map, output_path=os.path.join(plots_dir, "overall.png"))

    print("Filtering and saving subsets...")
    filter_and_save_subsets(df, processed_dir, labels_map)
    
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
