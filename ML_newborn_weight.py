import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, classification_report)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE


warnings.filterwarnings('ignore')

# 1 First we load data
train_df = pd.read_csv('us_births.csv')
test_df  = pd.read_csv('us_births_test.csv')


# 2 preprocessing data so it can be read, also removing rows where we dont have weight data for the newborn
def readable_data(df):
    df.dropna(subset=['newborn_birth_weight'], inplace=True)
    df['mother_diabetes_gestational'] = df['mother_diabetes_gestational'].map({'Y':1,'N':0})
    df['mother_risk_factor']          = df['mother_risk_factor'].astype(int)
    df['newborn_sex']                 = df['newborn_sex'].map({'M':1,'F':0})
    df['month_sin'] = np.sin(2*np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2*np.pi * df['month'] / 12)
    return df

# 3 applying the changes to datasets
train_df = readable_data(train_df)
test_df  = readable_data(test_df)

# 4 Then we engineer the features of the training and testing set, emphasis on gestation week, weight, height, cigarettes
def engineer(df):
    df['gest_sq']    = df['gestation_week']**2
    bins = [0, 30, 36, 40, 100]
    df['gest_bin'] = pd.cut(df['gestation_week'], bins=bins, labels=[0,1,2,3]).astype(int)
    df['gest_cat'] = pd.cut(df['gestation_week'], bins=bins,
                             labels=['<30','30-36','37-40','>40']).astype(str)
    df['frame_idx'] = df['mother_weight_delivery'] / df['mother_height']

    df['cig_total'] = df[[ 
        'daily_cigarette_prepregnancy', 
        'daily_cigarette_trimester_1', 
        'daily_cigarette_trimester_2', 
        'daily_cigarette_trimester_3'
    ]].sum(axis=1)
    df['any_cig']   = (df['cig_total'] > 0).astype(int)
    df['cig_heavy'] = (df['cig_total'] > 10).astype(int)
    
    return df

# 5 we apply the same function to both the training and test sets separately
train_df = engineer(train_df)
test_df = engineer(test_df)

# 6 we define features for future use, so we get feature and target values
features = [
    'month_sin','month_cos','mother_age','prenatal_care_starting_month',
    'mother_height','mother_bmi','mother_weight_prepregnancy','mother_weight_delivery',
    'mother_diabetes_gestational','newborn_sex','mother_risk_factor',
    'gest_sq','gest_bin','gest_cat','frame_idx','cig_total','any_cig','cig_heavy'
]
X_train, y_train = train_df[features], train_df['newborn_birth_weight']
X_test,  y_test  = test_df[features], test_df['newborn_birth_weight']

# 7 one hot encoding and preprocessing for different types of data, filling missing numeric values with median, standardizing numerical data etc.

num_cols = [c for c in features if c != 'gest_cat']
cat_cols = ['gest_cat']

# 8 making pipelines
numeric_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly',    PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)),
    ('scaler',  StandardScaler())
])

cat_tf = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer([
    ('num', numeric_tf, num_cols),
    ('cat', cat_tf,    cat_cols)
])

# 9 getting shap to work so we can see what features are important
def explain_with_shap(pipeline, X_raw, model_name='Model'):
    try:
        model = pipeline.named_steps['model'].regressor_
        X_trans = pipeline[:-1].transform(X_raw)
        explainer = shap.Explainer(model, X_trans)
        shap_vals = explainer(X_trans)
        shap.summary_plot(
            shap_vals.values, X_trans,
            feature_names=pipeline[:-1].get_feature_names_out(),
            show=True
        )
    except Exception as rip: #because too many errors to quit code here
        print(f"[{model_name}] SHAP doesn't work :():", rip)


# 10 creating our linear regression
regressor = TransformedTargetRegressor(
    regressor=LinearRegression(fit_intercept=True),  # Explicitly set if you want
    func=np.log1p,
    inverse_func=np.expm1
)

pipeline = Pipeline([
    ('preproc',    preprocessor),
    ('var_thresh', VarianceThreshold(threshold=0.01)),
    ('model',      regressor)
])

weights = np.where(y_train < 2500, 3, 1)

# 11 performing cross validation
cv = cross_validate(
    pipeline, X_train, y_train, cv=5,
    scoring=('neg_root_mean_squared_error','neg_mean_absolute_error')
)
cv_rmse = -cv['test_neg_root_mean_squared_error']
cv_mae  = -cv['test_neg_mean_absolute_error']
print(f"Linear regression training RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
print(f"Linear regression training MAE : {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")

# 12 putting the final model through the training set
pipeline.fit(X_train, y_train, model__sample_weight=weights)


# 13 evaluating how well we did
y_pred = pipeline.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mae    = mean_absolute_error(y_test, y_pred)
print(f"Linear test RMSE: {rmse:.2f}")
print(f"Linear test MAE: {mae:.2f}")       

# 14 a few plots to show what our model did
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.title("Linear model true vs. predicted birth weight")
plt.xlabel("True (g)"); plt.ylabel("Predicted (g)")
plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.histplot(y_test - y_pred, bins=40, kde=True)
plt.title("Linear model true vs. predicted weight residuals")
plt.xlabel("Error (g)"); plt.grid(True)
plt.tight_layout(); plt.show()

# 15 using shap to see what features affected the model most
explain_with_shap(pipeline, X_test, model_name="Linear")


# 16 making our random forests pipeline

rf_model = TransformedTargetRegressor(
    regressor=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    func=np.log1p,
    inverse_func=np.expm1
)
rf_pipeline = Pipeline([
    ('preproc',    preprocessor),
    ('var_thresh', VarianceThreshold(threshold=0.01)),
    ('model',      rf_model)
])

rf_pipeline.fit(X_train, y_train, model__sample_weight=weights)

rf_cv = cross_validate(
    rf_pipeline, X_train, y_train, cv=5,
    scoring=('neg_root_mean_squared_error','neg_mean_absolute_error')
)
rf_rmse = -rf_cv['test_neg_root_mean_squared_error']
rf_mae  = -rf_cv['test_neg_mean_absolute_error']
print(f"Random forest training RMSE: {rf_rmse.mean():.2f} ± {rf_rmse.std():.2f}")
print(f"Random forest training MAE : {rf_mae.mean():.2f} ± {rf_mae.std():.2f}")

rf_pred = rf_pipeline.predict(X_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_test_mae  = mean_absolute_error(y_test, rf_pred)
print(f"Random forest test RMSE: {rf_test_rmse:.2f}")
print(f"Random forest test MAE: {rf_test_mae:.2f}")    

# 17 drawing a plot of results
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=rf_pred, alpha=0.6)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.title("Random forest true vs. predicted birth weight")
plt.xlabel("True (g)"); plt.ylabel("Pred (g)")
plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.histplot(y_test - rf_pred, bins=40, kde=True)
plt.title("Random forest true vs. predicted birth weight residuals)")
plt.xlabel("Error (g)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 18 and finally my favorite, shap
explain_with_shap(rf_pipeline, X_test, model_name="Random Forest")


# 19 testing how classification would go, target for low baby weight LBW being < 2500
y_bin_train = (y_train < 2500).astype(int)
y_bin_test = (y_test < 2500).astype(int)

# 20 our pipeline, preprocessing smote and random forest
clf = ImbPipeline([
    ('preproc', preprocessor),
    ('smote', BorderlineSMOTE(k_neighbors=3, random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 21 using stratifiedkfold
strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 22 cross validating Cross-validation
cv_class = cross_validate(clf, X_train, y_bin_train, cv=strat_kf, scoring='f1', return_train_score=True)
print(f"\nClassifier cross validation F1 score: {cv_class['test_score'].mean():.2f} ± {cv_class['test_score'].std():.2f}")

# 23 final training
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# 24 applying SMOTE because we struggle with LW babies
smote = BorderlineSMOTE(k_neighbors=2, sampling_strategy=0.5, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_proc, y_bin_train)

# 25 model fitting
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_smote, y_smote)

# 26 predictions and evaluations
y_probs = clf_model.predict_proba(X_test_proc)[:, 1]
y_bin_pred = (y_probs >= 0.3).astype(int)

# 27 getting precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_bin_test, y_probs)

# 28 calculating f1 scores for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# 29 gettin best treshold
best_thresh = thresholds[np.argmax(f1_scores)]
y_bin_pred_best_thresh = (y_probs >= best_thresh).astype(int)

# 30 printing our confusion matrix
cm = confusion_matrix(y_bin_test, y_bin_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal weight", "Low weight"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.show()

# 31 plot for our precision-recall curve
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision-recall curve")
plt.tight_layout()
plt.show()

print("\n--- Classification Report ---")
print(classification_report(y_bin_test, y_bin_pred, target_names=["Normal", "Low birth weight"]))