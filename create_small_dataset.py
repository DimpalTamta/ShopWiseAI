import pandas as pd

# Path to your large CSV (the original 1.5M file)
LARGE_CSV = "amazon_products.csv"  # or the actual filename
SMALL_CSV = "amazon_products_5000.csv"

# Read only 5,000 random rows
df = pd.read_csv(LARGE_CSV)
sample = df.sample(n=5000, random_state=42)  # 42 ensures reproducibility

# Save to a new CSV
sample.to_csv(SMALL_CSV, index=False)
print(f"✅ Created {SMALL_CSV} with {len(sample)} rows")