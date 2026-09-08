from pyzotero import zotero
import pandas as pd
import re
import sys

class ZoteroFetcher:
    def __init__(self, library_id, library_type, api_key):
        """
        Initialize Zotero API client.
        library_id: Numeric User ID or Group ID.
        library_type: 'user' or 'group'
        """
        self.zot = zotero.Zotero(library_id, library_type, api_key)

    def get_collection_id_by_name(self, name):
        """Find a collection ID by its name."""
        try:
            collections = self.zot.collections()
            for col in collections:
                if col['data']['name'].lower() == name.lower():
                    return col['key']
            
            # Not found, let's list available ones for the error message
            available = [col['data']['name'] for col in collections]
            return available # Return list if not found
        except Exception as e:
            raise e

    def fetch_items(self, collection_name=None):
        """Fetch items, optionally filtered by collection name."""
        try:
            if collection_name:
                result = self.get_collection_id_by_name(collection_name)
                if isinstance(result, list):
                    # It's the list of available collections
                    print(f"Error: Collection '{collection_name}' not found.")
                    print(f"Available collections: {', '.join(result) if result else 'None found'}")
                    sys.exit(1)
                
                print(f"Fetching items from collection: {collection_name} ({result})")
                items = self.zot.everything(self.zot.collection_items(result))
            else:
                print("Fetching all top-level items from library...")
                items = self.zot.everything(self.zot.top())
            
            return items
        except Exception as e:
            # Check for common "Invalid user ID" error which happens if library_id is wrong
            err_msg = str(e)
            if "Invalid user ID" in err_msg or "400" in err_msg:
                print("\nError: Invalid Zotero User ID or Library ID.")
                print("Please ensure 'library_id' in config.yaml is your NUMERIC Zotero ID.")
                print("You can find your numeric ID at https://www.zotero.org/settings/keys")
            raise e

    def to_dataframe(self, items):
        """Convert Zotero items to a pandas DataFrame compatible with the pipeline."""
        processed_data = []
        
        for item in items:
            data = item.get('data', {})
            if 'title' not in data:
                continue
                
            full_date = data.get('date', '')
            year = None
            if full_date:
                match = re.search(r'\d{4}', full_date)
                if match:
                    year = int(match.group())

            processed_data.append({
                'Title': data.get('title', ''),
                'Abstract Note': data.get('abstractNote', ''),
                'Publication Year': year,
                'Source': 'Zotero',
                'Item Type': data.get('itemType', ''),
                'Key': data.get('key', '')
            })
            
        df = pd.DataFrame(processed_data)
        if not df.empty:
            df = df[df['Abstract Note'].str.strip() != ''].copy()
        
        return df
