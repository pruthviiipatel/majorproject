# GENOMICS DATA INTEGRATION GUIDE
# For Lung Tumor Diagnosis and Drug Discovery Project

## 📁 How to Set Up Your Genomics Dataset

### Step 1: Download Your Data
1. Go to: https://drive.google.com/drive/folders/18-PfaDpNFHAXr3sKkdJH1h8sSg5LJSsv?usp=sharing
2. Download all files to your local machine
3. Place them in a folder called `genomics_data/`

### Step 2: Update the Notebook
In Cell 3 of the notebook, update the path:
```python
GENOMICS_DATA = "/path/to/genomics_data/your_main_file.csv"
```

---

## 🧬 Common Genomics Data Formats and How to Handle Them

### Format 1: Gene Expression Matrix
**Expected Structure:**
```
patient_id, gene_1, gene_2, gene_3, ..., gene_n, drug_response
P001, 5.23, 3.45, 7.89, ..., 2.34, Cisplatin
P002, 4.56, 6.78, 5.43, ..., 3.21, Pembrolizumab
```

**Code to Load:**
```python
import pandas as pd
genomics_df = pd.read_csv('genomics_data/gene_expression.csv')
print(genomics_df.head())
print(f"Shape: {genomics_df.shape}")
```

### Format 2: Mutation Data
**Expected Structure:**
```
patient_id, gene, mutation_type, variant_allele_frequency
P001, EGFR, missense, 0.45
P001, KRAS, nonsense, 0.32
P002, TP53, frameshift, 0.67
```

**Code to Load:**
```python
mutation_df = pd.read_csv('genomics_data/mutations.csv')
# Pivot to create patient-level features
mutation_pivot = mutation_df.pivot_table(
    index='patient_id',
    columns='gene',
    values='variant_allele_frequency',
    fill_value=0
)
```

### Format 3: Clinical + Genomics Combined
**Expected Structure:**
```
patient_id, age, stage, EGFR_expr, KRAS_expr, ..., treatment, response
P001, 65, III, 5.2, 3.4, ..., Cisplatin, Complete
P002, 58, IV, 4.1, 6.7, ..., Nivolumab, Partial
```

---

## 💊 Drug Discovery Approaches Based on Your Data

### Approach 1: Supervised Drug Recommendation
**When you have:** Patient genomics + Known effective drugs

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Prepare data
X = genomics_df.drop(['patient_id', 'effective_drug'], axis=1)
y = genomics_df['effective_drug']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train_scaled, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### Approach 2: Biomarker Discovery
**When you want to:** Identify genes that predict drug response

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Get top genes
feature_scores = pd.DataFrame({
    'gene': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("Top 20 Biomarkers for Drug Response:")
print(feature_scores.head(20))
```

### Approach 3: Drug-Gene Interaction Analysis
**When you want to:** Understand which genes affect specific drugs

```python
# For each drug, find associated genes
drugs = genomics_df['effective_drug'].unique()

for drug in drugs:
    # Create binary target: this drug vs others
    y_drug = (genomics_df['effective_drug'] == drug).astype(int)
    
    # Find important genes for this drug
    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(X, y_drug)
    
    top_genes = X.columns[selector.get_support()]
    print(f"\nTop genes for {drug}:")
    print(top_genes.tolist())
```

### Approach 4: Clustering-Based Drug Discovery
**When you want to:** Group similar patients for personalized treatment

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Dimensionality reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Analyze drug response per cluster
genomics_df['cluster'] = clusters
cluster_drugs = genomics_df.groupby('cluster')['effective_drug'].value_counts()
print(cluster_drugs)
```

---

## 🔬 Integration with Tumor Imaging Features

### Combine Imaging + Genomics for Better Predictions

```python
# Tumor features from CT scan (from Cell 16 of main notebook)
tumor_features = extract_tumor_features(image, predicted_mask)

# Convert to DataFrame
tumor_df = pd.DataFrame([tumor_features])

# Merge with genomics
patient_id = 'P001'  # Match to your patient
genomic_features = genomics_df[genomics_df['patient_id'] == patient_id].drop('patient_id', axis=1)

# Combine
combined_features = pd.concat([tumor_df.reset_index(drop=True), 
                               genomic_features.reset_index(drop=True)], axis=1)

# Now use combined features for prediction
predicted_drug = drug_model.predict(scaler.transform(combined_features))
print(f"Recommended drug based on imaging + genomics: {predicted_drug[0]}")
```

---

## 📈 Advanced Analysis Options

### 1. Gene Set Enrichment Analysis (GSEA)
```python
# If you have gene expression data
from scipy.stats import ranksums

# Compare gene expression between responders and non-responders
responders = genomics_df[genomics_df['response'] == 'Complete']
non_responders = genomics_df[genomics_df['response'] == 'Partial']

gene_columns = [col for col in genomics_df.columns if col.startswith('gene_')]

pvalues = {}
for gene in gene_columns:
    stat, pval = ranksums(responders[gene], non_responders[gene])
    pvalues[gene] = pval

# Find significantly different genes
significant_genes = {k: v for k, v in pvalues.items() if v < 0.05}
print(f"Found {len(significant_genes)} significantly different genes")
```

### 2. Survival Analysis (if you have time-to-event data)
```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

# For each drug, plot survival curves
for drug in drugs:
    drug_patients = genomics_df[genomics_df['effective_drug'] == drug]
    kmf.fit(drug_patients['survival_time'], drug_patients['event_occurred'])
    kmf.plot_survival_function(label=drug)

plt.title('Survival Curves by Drug')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()
```

### 3. Deep Learning for Drug Discovery
```python
import torch
import torch.nn as nn

class DrugPredictionNN(nn.Module):
    def __init__(self, input_dim, num_drugs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_drugs)
        )
    
    def forward(self, x):
        return self.network(x)

# Train the model
model = DrugPredictionNN(input_dim=X.shape[1], num_drugs=len(drugs))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train_scaled))
    loss = criterion(outputs, torch.LongTensor(y_train_encoded))
    loss.backward()
    optimizer.step()
```

---

## 🎯 What Information I Need From You

To customize the code specifically for your dataset, please provide:

1. **File name(s)** in your Google Drive folder
2. **First 5-10 rows** of your main dataset (copy-paste or screenshot)
3. **Column names** and what they represent
4. **Target variable** - What you want to predict (drug name? response? survival?)
5. **Data dictionary** - If you have documentation about the dataset

Once you provide this information, I'll:
- ✅ Create custom data loading code
- ✅ Build appropriate feature engineering pipeline
- ✅ Design the best model architecture for your data
- ✅ Integrate seamlessly with the tumor imaging analysis
- ✅ Generate comprehensive drug recommendations

---

## 📝 Quick Start Checklist

- [ ] Download genomics data from Google Drive
- [ ] Identify file format (CSV, Excel, TSV, etc.)
- [ ] Check column names and data types
- [ ] Identify target variable for drug prediction
- [ ] Update GENOMICS_DATA path in Cell 3 of main notebook
- [ ] Run Cells 18-22 with your actual data
- [ ] Analyze results and refine model

---

## 💡 Tips for Best Results

1. **Data Quality**: Check for missing values, outliers
2. **Feature Scaling**: Always scale gene expression data
3. **Class Imbalance**: If some drugs are rare, use balanced sampling
4. **Validation**: Use cross-validation for reliable estimates
5. **Interpretability**: Use feature importance to understand predictions
6. **Clinical Validation**: Always validate with domain experts

---

Ready to proceed! Share your dataset details and I'll create the perfect solution for your project.
