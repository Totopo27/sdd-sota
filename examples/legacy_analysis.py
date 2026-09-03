import os
import pandas as pd
import argparse
import sys
from dotenv import load_dotenv
from litreview.analysis import unify_databases, clean_dataframe, run_classification_pipeline
from litreview.utils import load_config
from litreview.zotero import ZoteroFetcher

# Load .env if present (gitignored; .env-example is the template)
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run literature review classification pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--fetch-zotero", action="store_true", help="Fetch data from Zotero library instead of local CSVs")
    args = parser.parse_args()

    config = load_config(args.config)
    
    paths = config.get("paths", {})
    raw_dir = paths.get("raw_data", "data/raw")
    processed_dir = paths.get("processed_data", "data/processed")
    
    sources = config.get("sources", [])
    models = config.get("models", [])
    labels_map = config.get("labels", {})

    if args.fetch_zotero:
        lib_id = os.environ.get("ZOTERO_LIBRARY_ID")
        lib_type = config.get("zotero", {}).get("library_type", "user")
        api_key = os.environ.get("ZOTERO_API_KEY")
        col_name = config.get("zotero", {}).get("collection_name")

        # If library_id looks like a name (not numeric), warn user
        if lib_id and not str(lib_id).isdigit():
            print(f"Warning: ZOTERO_LIBRARY_ID '{lib_id}' does not look like a numeric Zotero ID.")
            print("Zotero Library IDs are usually numeric (e.g. 1234567).")

        if not lib_id or not api_key:
            print("Error: ZOTERO_LIBRARY_ID and ZOTERO_API_KEY must be set in .env when using --fetch-zotero")
            print("Copy .env-example to .env and fill in your credentials.")
            return
            
        try:
            fetcher = ZoteroFetcher(lib_id, lib_type, api_key)
            items = fetcher.fetch_items(collection_name=col_name)
            df = fetcher.to_dataframe(items)
            print(f"Fetched {len(df)} items with abstracts from Zotero.")
        except Exception as e:
            # Errors are already printed in fetcher.fetch_items
            sys.exit(1)
    else:
        # 1. Unify from local CSVs
        print(f"Unifying databases from {raw_dir}...")
        df = unify_databases(sources, data_dir=raw_dir)
        
    if df.empty:
        print(f"No data found. Please check your config, Zotero credentials, or data files.")
        return

    # 2. Clean
    print("Cleaning data...")
    df = clean_dataframe(df)

    # 3. Classify optimized
    print("Running optimized classification pipeline...")
    df = run_classification_pipeline(df, models, labels_map)

    # 4. Save
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "classified.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
