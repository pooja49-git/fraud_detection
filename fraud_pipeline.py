# import argparse
# import logging
# import sys
# import os
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when, lit, udf
# from pyspark.sql.types import DoubleType
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA
# from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.clustering import KMeans
# from pyspark.ml.linalg import Vectors

# # Try to import XGBoost
# try:
#     from sparkxgb import XGBoostClassifier
#     xgb_available = True
# except ImportError:
#     xgb_available = False

# # -------------------------
# # Logging Setup
# # -------------------------
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# log = logging.getLogger("fraud_pipeline")

# # -------------------------
# # Parse Arguments
# # -------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, default="/scratch/m24csa020/Fraud_Detection_Pyspark/PS_20174392719_1491204439457_log.csv")
# parser.add_argument("--seed", type=int, default=42)
# args = parser.parse_args()

# input_path = args.input
# seed = args.seed

# # -------------------------
# # Spark Session
# # -------------------------
# spark = SparkSession.builder.appName("FraudDetectionPipeline").getOrCreate()

# # -------------------------
# # Load Data with Check
# # -------------------------
# if not os.path.exists(input_path):
#     log.error(f"Input file not found: {input_path}")
#     sys.exit(1)

# log.info(f"Loading data from {input_path}")
# df = spark.read.csv(input_path, header=True, inferSchema=True)
# df = df.withColumn("isFraud", col("isFraud").cast("double"))
# log.info(f"Total dataset count: {df.count()}")

# # -------------------------
# # Balance Classes
# # -------------------------
# fraud_count = df.filter(col("isFraud") == 1.0).count()
# nonfraud_count = df.filter(col("isFraud") == 0.0).count()
# ratio = fraud_count / nonfraud_count if nonfraud_count != 0 else 0
# log.info(f"Fraud={fraud_count}, Non-fraud={nonfraud_count}, Ratio={ratio:.6f}")

# nonfraud_df = df.filter(col("isFraud") == 0.0).sample(withReplacement=False, fraction=min(ratio*3, 1.0), seed=seed)
# fraud_df = df.filter(col("isFraud") == 1.0)
# balanced_df = nonfraud_df.union(fraud_df).repartition(200).cache()
# log.info(f"Balanced dataset size: {balanced_df.count()}")

# # -------------------------
# # Train/Test Split
# # -------------------------
# fractions = {0.0: 0.8, 1.0: 0.8}
# train_df = balanced_df.sampleBy("isFraud", fractions, seed=seed).cache()
# test_df = balanced_df.subtract(train_df).cache()
# log.info(f"Train={train_df.count()}, Test={test_df.count()}")

# # -------------------------
# # Preprocessing (cached)
# # -------------------------
# type_indexer = StringIndexer(inputCol="type", outputCol="type_index")
# type_encoder = OneHotEncoder(inputCols=["type_index"], outputCols=["type_ohe"])
# feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_ohe"]
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
# scaler = StandardScaler(inputCol="features_raw", outputCol="features")
# preproc_stages = [type_indexer, type_encoder, assembler, scaler]

# # Preprocess train and test once and cache
# preproc_pipeline = Pipeline(stages=preproc_stages)
# preproc_model = preproc_pipeline.fit(train_df)
# train_preproc = preproc_model.transform(train_df).cache()
# test_preproc = preproc_model.transform(test_df).cache()
# train_preproc.count()  # Force evaluation
# test_preproc.count()

# # -------------------------
# # Evaluation Function
# # -------------------------
# results = {}

# def evaluate_model(predictions, label="isFraud", name="Model"):
#     tp = predictions.filter((col(label) == 1.0) & (col("prediction") == 1.0)).count()
#     tn = predictions.filter((col(label) == 0.0) & (col("prediction") == 0.0)).count()
#     fp = predictions.filter((col(label) == 0.0) & (col("prediction") == 1.0)).count()
#     fn = predictions.filter((col(label) == 1.0) & (col("prediction") == 0.0)).count()

#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#     acc = (tp + tn) / (tp + tn + fp + fn)

#     # Handle missing rawPrediction gracefully
#     if "rawPrediction" in predictions.columns:
#         roc_auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)
#         pr_auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(predictions)
#     else:
#         roc_auc = None
#         pr_auc = None

#     results[name] = {
#         "Accuracy": acc,
#         "Precision": precision,
#         "Recall": recall,
#         "F1": f1,
#         "ROC-AUC": roc_auc,
#         "PR-AUC": pr_auc
#     }

#     log.info(f"=== {name} Evaluation ===")
#     log.info(f"Accuracy={acc:.6f}, Precision={precision:.6f}, Recall={recall:.6f}, F1={f1:.6f}, ROC-AUC={roc_auc}, PR-AUC={pr_auc}")

# # -------------------------
# # Models to Run
# # -------------------------
# models_to_run = ["lr", "rf", "anomaly"]
# if xgb_available:
#     models_to_run.append("xgb")
# else:
#     log.warning("XGBoost not available; install sparkxgb to enable it.")

# # -------------------------
# # Train & Evaluate Models
# # -------------------------
# best_predictions = None
# best_model_name = None
# best_f1 = 0
# best_model_obj = None

# for choice in models_to_run:
#     if choice == "lr":
#         log.info("=== Training Logistic Regression ===")
#         lr = LogisticRegression(featuresCol="features", labelCol="isFraud", maxIter=10)
#         lr_model = lr.fit(train_preproc)
#         preds = lr_model.transform(test_preproc)
#         evaluate_model(preds, name="LogisticRegression")
#         if results["LogisticRegression"]["F1"] > best_f1:
#             best_f1 = results["LogisticRegression"]["F1"]
#             best_predictions = preds
#             best_model_name = "LogisticRegression"
#             best_model_obj = lr_model

#     elif choice == "rf":
#         log.info("=== Training Random Forest ===")
#         rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", numTrees=50, maxDepth=10, seed=seed)
#         rf_model = rf.fit(train_preproc)
#         preds = rf_model.transform(test_preproc)
#         evaluate_model(preds, name="RandomForest")
#         if results["RandomForest"]["F1"] > best_f1:
#             best_f1 = results["RandomForest"]["F1"]
#             best_predictions = preds
#             best_model_name = "RandomForest"
#             best_model_obj = rf_model

#     elif choice == "xgb":
#         log.info("=== Training XGBoost ===")
#         xgb = XGBoostClassifier(featuresCol="features", labelCol="isFraud", missing=0.0, eta=0.1, maxDepth=6,
#                                 objective="binary:logistic", numRound=50, numWorkers=4, seed=seed)
#         xgb_model = xgb.fit(train_preproc)
#         preds = xgb_model.transform(test_preproc)
#         evaluate_model(preds, name="XGBoost")
#         if results["XGBoost"]["F1"] > best_f1:
#             best_f1 = results["XGBoost"]["F1"]
#             best_predictions = preds
#             best_model_name = "XGBoost"
#             best_model_obj = xgb_model

#     elif choice == "anomaly":
#         log.info("=== Training Anomaly Detector (PCA + KMeans) ===")
#         pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
#         kmeans = KMeans(k=2, featuresCol="pcaFeatures", seed=seed)
#         anomaly_pipeline = Pipeline(stages=[pca, kmeans])
#         anomaly_model = anomaly_pipeline.fit(train_preproc)
#         preds = anomaly_model.transform(test_preproc)

#         fraud_majority = preds.groupBy("prediction").agg({"isFraud": "avg"}).collect()
#         fraud_map = {r["prediction"]: (1 if r["avg(isFraud)"] >= 0.5 else 0) for r in fraud_majority}

#         fraud_clusters = [c for c, label in fraud_map.items() if label == 1]
#         fraud_cluster = fraud_clusters[0] if len(fraud_clusters) > 0 else 0

#         fraud_center = anomaly_model.stages[-1].clusterCenters()[fraud_cluster]
#         def distance_to_fraud(features):
#             return float(Vectors.dense(features).squared_distance(Vectors.dense(fraud_center)))
#         distance_udf = udf(distance_to_fraud, DoubleType())
#         preds = preds.withColumn("rawPrediction", -distance_udf("pcaFeatures"))
#         preds = preds.withColumn("prediction", when(col("prediction") == lit(fraud_cluster), lit(1)).otherwise(lit(0)))

#         evaluate_model(preds, name="AnomalyDetector")
#         if results["AnomalyDetector"]["F1"] > best_f1:
#             best_f1 = results["AnomalyDetector"]["F1"]
#             best_predictions = preds
#             best_model_name = "AnomalyDetector"
#             best_model_obj = anomaly_model

# # -------------------------
# # Print Summary
# # -------------------------
# if results:
#     print("\n=== Summary of All Models ===")
#     for model_name, metrics in results.items():
#         print(f"\n{model_name}:")
#         for k, v in metrics.items():
#             print(f"  {k}: {v}")

# # -------------------------
# # Save Best Model
# # -------------------------
# if best_model_obj:
#     log.info(f"Saving best model: {best_model_name}")
#     model_path = f"./best_{best_model_name}_model"
#     if os.path.exists(model_path):
#         import shutil
#         shutil.rmtree(model_path)
#     best_model_obj.save(model_path)

# spark.stop()
import argparse
import logging
import sys
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

# Try to import XGBoost
try:
    from sparkxgb import XGBoostClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("fraud_pipeline")

# -------------------------
# Parse Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/scratch/m24csa020/Fraud_Detection_Pyspark/PS_20174392719_1491204439457_log.csv")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

input_path = args.input
seed = args.seed

# -------------------------
# Spark Session
# -------------------------
spark = SparkSession.builder.appName("FraudDetectionPipeline").getOrCreate()

# -------------------------
# Load Data with Check
# -------------------------
if not os.path.exists(input_path):
    log.error(f"Input file not found: {input_path}")
    sys.exit(1)

log.info(f"Loading data from {input_path}")
df = spark.read.csv(input_path, header=True, inferSchema=True)
df = df.withColumn("isFraud", col("isFraud").cast("double"))
log.info(f"Total dataset count: {df.count()}")

# -------------------------
# Balance Classes
# -------------------------
fraud_count = df.filter(col("isFraud") == 1.0).count()
nonfraud_count = df.filter(col("isFraud") == 0.0).count()
ratio = fraud_count / nonfraud_count if nonfraud_count != 0 else 0
log.info(f"Fraud={fraud_count}, Non-fraud={nonfraud_count}, Ratio={ratio:.6f}")

nonfraud_df = df.filter(col("isFraud") == 0.0).sample(withReplacement=False, fraction=min(ratio*3, 1.0), seed=seed)
fraud_df = df.filter(col("isFraud") == 1.0)
balanced_df = nonfraud_df.union(fraud_df).repartition(200).cache()
log.info(f"Balanced dataset size: {balanced_df.count()}")

# -------------------------
# Train/Test Split
# -------------------------
fractions = {0.0: 0.8, 1.0: 0.8}
train_df = balanced_df.sampleBy("isFraud", fractions, seed=seed).cache()
test_df = balanced_df.subtract(train_df).cache()
log.info(f"Train={train_df.count()}, Test={test_df.count()}")

# -------------------------
# Preprocessing (cached)
# -------------------------
type_indexer = StringIndexer(inputCol="type", outputCol="type_index")
type_encoder = OneHotEncoder(inputCols=["type_index"], outputCols=["type_ohe"])
feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_ohe"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
preproc_stages = [type_indexer, type_encoder, assembler, scaler]

preproc_pipeline = Pipeline(stages=preproc_stages)
preproc_model = preproc_pipeline.fit(train_df)
train_preproc = preproc_model.transform(train_df).cache()
test_preproc = preproc_model.transform(test_df).cache()
train_preproc.count()
test_preproc.count()

# -------------------------
# Evaluation Function
# -------------------------
results = {}
all_metrics = {}

def evaluate_model(predictions, label="isFraud", name="Model"):
    tp = predictions.filter((col(label) == 1.0) & (col("prediction") == 1.0)).count()
    tn = predictions.filter((col(label) == 0.0) & (col("prediction") == 0.0)).count()
    fp = predictions.filter((col(label) == 0.0) & (col("prediction") == 1.0)).count()
    fn = predictions.filter((col(label) == 1.0) & (col("prediction") == 0.0)).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)

    if "rawPrediction" in predictions.columns:
        roc_auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)
        pr_auc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(predictions)
    else:
        roc_auc, pr_auc = None, None

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }
    results[name] = metrics
    all_metrics[name] = metrics

    log.info(f"=== {name} Evaluation ===")
    log.info(metrics)

# -------------------------
# Models to Run
# -------------------------
models_to_run = ["lr", "rf", "anomaly"]
if xgb_available:
    models_to_run.append("xgb")
else:
    log.warning("XGBoost not available; install sparkxgb to enable it.")

best_model = None
best_predictions = None
best_name = None
best_f1 = -1

# -------------------------
# Train & Evaluate Models
# -------------------------
for choice in models_to_run:
    if choice == "lr":
        lr = LogisticRegression(featuresCol="features", labelCol="isFraud", maxIter=10)
        model = lr.fit(train_preproc)
        preds = model.transform(test_preproc)
        evaluate_model(preds, name="LogisticRegression")

    elif choice == "rf":
        rf = RandomForestClassifier(featuresCol="features", labelCol="isFraud", numTrees=50, maxDepth=10, seed=seed)
        model = rf.fit(train_preproc)
        preds = model.transform(test_preproc)
        evaluate_model(preds, name="RandomForest")

    elif choice == "xgb":
        xgb = XGBoostClassifier(featuresCol="features", labelCol="isFraud",
                                missing=0.0, eta=0.1, maxDepth=6,
                                objective="binary:logistic", numRound=50,
                                numWorkers=4, seed=seed)
        model = xgb.fit(train_preproc)
        preds = model.transform(test_preproc)
        evaluate_model(preds, name="XGBoost")

        # Track best
        f1 = results["XGBoost"]["F1"]
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_predictions = preds
            best_name = "XGBoost"

    elif choice == "anomaly":
        pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
        kmeans = KMeans(k=2, featuresCol="pcaFeatures", seed=seed)
        anomaly_pipeline = Pipeline(stages=[pca, kmeans])
        model = anomaly_pipeline.fit(train_preproc)
        preds = model.transform(test_preproc)

        fraud_majority = preds.groupBy("prediction").agg({"isFraud": "avg"}).collect()
        fraud_map = {r["prediction"]: (1 if r["avg(isFraud)"] >= 0.5 else 0) for r in fraud_majority}
        fraud_clusters = [c for c, label in fraud_map.items() if label == 1]
        fraud_cluster = fraud_clusters[0] if len(fraud_clusters) > 0 else 0

        fraud_center = model.stages[-1].clusterCenters()[fraud_cluster]
        def distance_to_fraud(features):
            return float(Vectors.dense(features).squared_distance(Vectors.dense(fraud_center)))
        distance_udf = udf(distance_to_fraud, DoubleType())
        preds = preds.withColumn("rawPrediction", -distance_udf("pcaFeatures"))
        preds = preds.withColumn("prediction", when(col("prediction") == lit(fraud_cluster), lit(1)).otherwise(lit(0)))

        evaluate_model(preds, name="AnomalyDetector")

# -------------------------
# Save Best Model & Metrics
# -------------------------
if best_model and best_name == "XGBoost":
    log.info("Saving best XGBoost model for Streamlit...")

    # Save Spark ML model (always works)
    best_model.write().overwrite().save("best_XGBoost_model")

    saved_native = False
    try:
        # Try to get the native booster safely
        booster = None
        if hasattr(best_model, "nativeBooster") and best_model.nativeBooster is not None:
            booster = best_model.nativeBooster
        elif hasattr(best_model, "get_booster"):  # newer API
            booster = best_model.get_booster()

        if booster is not None:
            booster.save_model("xgb_model.json")
            log.info("✅ Saved xgb_model.json successfully")
            saved_native = True
        else:
            log.warning("⚠️ Native booster not found in this XGBoost model.")

    except Exception as e:
        log.warning(f"⚠️ Could not save xgb_model.json due to: {e}")

    # Save metrics
    with open("results.json", "w") as f:
        json.dump({
            "best_model": "xgb_model.json" if saved_native else "best_XGBoost_model",
            "all_metrics": all_metrics
        }, f, indent=4)

    log.info("✅ Saved results.json with all metrics")

# -------------------------
# Print Summary
# -------------------------
if results:
    print("\n=== Summary of All Models ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

spark.stop()





