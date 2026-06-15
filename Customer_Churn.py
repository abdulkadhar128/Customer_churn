# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score,precision_score,recall_score
# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# DATA UNDERSTANDING 
df = pd.read_csv("customer_churn_dataset.csv")
print(df.head())
  
print(f" Dataset Shape: {df.shape}")
print(f"\n Data Types:\n{df.dtypes}")
print(f"\n Missing Values:\n{df.isnull().sum()}")

# Data Cleaning
print("duplicate rows:", df.duplicated().sum())

print("Unique Values:", df.nunique())

# drop
df = df.drop("CustomerID", axis=1)
print(df.head())

# Labeling
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in ["Gender", "Subscription Type", "Contract Length"]:
    df[col] = le.fit_transform(df[col])

print(f"\n Data Types:\n{df.dtypes}")

#EDA
#Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

#Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data preprocessing
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mdoel Training

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Decision Tree

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Random Forest

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest")
print("Accuracy:",accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))




#Model Comparison
models = {
    "Logistic Regression":accuracy_score(y_test, y_pred_lr),

    "KNN":accuracy_score(y_test, y_pred_knn),

    "Decision Tree":accuracy_score(y_test, y_pred_dt),

    "Random Forest":accuracy_score(y_test, y_pred_rf)
}

for model, acc in models.items():
    print(model, ":", round(acc,4))
  
#Dumping
import joblib
joblib.dump(rf, "customer_churn_model.pkl")