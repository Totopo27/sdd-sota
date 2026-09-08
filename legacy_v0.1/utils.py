import re
import yaml

def normalize_text(text):
    """Normalize text for comparison by removing punctuation, lowercasing, and trimming whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\W+', '', text.lower().strip())

def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
