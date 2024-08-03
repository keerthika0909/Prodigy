import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('population_data.csv')
print(df.head())
print(df.columns)
if 'Year' not in df.columns or 'Population' not in df.columns:
    raise ValueError("The dataset does not have the required 'Year' and 'Population' columns.")
data = df[['Year', 'Population']]
data['Year'] = data['Year'].astype(str)  
data['Population'] = pd.to_numeric(data['Population'], errors='coerce')  
data = data.dropna(subset=['Year', 'Population'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Population', data=data, palette='viridis')
plt.title('Population Distribution Over Years')
plt.xlabel('Year')
plt.ylabel('Population')
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()
