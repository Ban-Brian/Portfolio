import zipfile
import pandas as pd

# Extract and read the dataset
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Load recipe file
df = pd.read_csv('data/RAW_recipes.csv')

# Inspect structure
df[['name', 'ingredients', 'calories']].head()