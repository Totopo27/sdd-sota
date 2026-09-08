import pandas as pd
import os
from datasets import Dataset
from .utils import normalize_text

def unify_databases(sources, data_dir="data/raw"):
    """Combine CSVs from multiple sources into a single DataFrame."""
    source_dfs = []

    for source in sources:
        filepath = os.path.join(data_dir, f"Monografia-{source}.csv")
        if os.path.exists(filepath):
            tmpdf = pd.read_csv(filepath)
            tmpdf["Source"] = source
            source_dfs.append(tmpdf)
        else:
            print(f"Warning: {filepath} not found.")
    
    if not source_dfs:
        return pd.DataFrame()

    df = pd.concat(source_dfs, axis=0)
    df = df.drop(df[df["Publication Year"] < 2000].index)
    df = df.drop(df[df["Publication Year"] > 2024].index)
    df.reset_index(drop=True, inplace=True)

    return df

def clean_dataframe(df):
    """Normalize titles and remove duplicates."""
    df["TitleNorm"] = df["Title"].map(normalize_text)
    df = df.drop_duplicates(subset=["TitleNorm"], keep="last")
    df.reset_index(drop=True, inplace=True)
    return df

def run_classification_pipeline(df, models, labels_map, classifier_kwargs=None):
    """
    Run multi-label zero-shot classification efficiently.
    
    Args:
        df: DataFrame with 'Abstract Note'
        models: List of model strings
        labels_map: Dict of {abbr: full_label}
        classifier_kwargs: Additional args for pipeline
    """
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    from tqdm import tqdm
    
    if classifier_kwargs is None:
        classifier_kwargs = {"device": 0, "torch_dtype": "auto"}

    abbrs = list(labels_map.keys())
    full_labels = [labels_map[abbr] for abbr in abbrs]
    
    dataset = Dataset.from_pandas(df[['Abstract Note']])

    for model_name in models:
        model_short = model_name.split("/")[-1]
        print(f"Initializing model: {model_name}")
        classifier = pipeline("zero-shot-classification", model=model_name, **classifier_kwargs)
        
        print(f"Running multi-label classification for {len(full_labels)} labels...")
        # Using KeyDataset and pipeline call for optimal batching
        results = classifier(
            KeyDataset(dataset, "Abstract Note"), 
            candidate_labels=full_labels, 
            multi_label=True, 
            batch_size=32
        )
        
        all_scores = []
        for res in tqdm(results, total=len(df), desc=f"Model: {model_short}"):
            # Map results back to the order of full_labels/abbrs
            # HF results are sorted by score, so we need a map
            score_map = dict(zip(res['labels'], res['scores']))
            all_scores.append([score_map[label] for label in full_labels])
        
        scores_df = pd.DataFrame(all_scores, columns=[f"{abbr}_{model_short}" for abbr in abbrs])
        df = pd.concat([df, scores_df], axis=1)
            
    return df
