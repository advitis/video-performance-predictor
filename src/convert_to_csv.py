import pandas as pd

# Replace with your actual file path
input_file = r"/Users/adviti/Documents/Documents Archive/Analysis Apr 2024/Initial Engagement v1.xlsx"
output_file = r"/Users/adviti/Documents/Documents Archive/Analysis Apr 2024/Initial Engagement v1.csv"

# Load the first sheet
df = pd.read_excel(input_file, sheet_name=0)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"✅ Converted to {output_file}")
