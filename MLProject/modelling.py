import pandas as pd
import os 
import mlflow 
import mlflow.sklearn
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, learning_curve 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from joblib import dump

dagshub_usn = 'ReynardAdimas'
dagshub_repo = 'SMSML_Reynard-Adimas-Nabil'

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "whatsapp_review_preprocessing", "train_processed.csv")
    test_path = os.path.join(base_path, "whatsapp_review_preprocessing", "test_processed.csv") 

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) 

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1] 

    return X_train, y_train, X_test, y_test 

def train():
    dagshub.init(repo_owner=dagshub_usn, repo_name=dagshub_repo, mlflow=True)
    mlflow.set_experiment("Eksperimen Basic")
    X_train, y_train, X_test, y_test = load_data()
    models_config = {
        "RandomForest":{
            "model" : RandomForestClassifier(random_state=42),
            "params" : {
                'n_estimators' : [50, 100], 
                'max_depth' : [10,20,None]
            } 
        }, 
        "DecisionTree" : {
            "model" : DecisionTreeClassifier(random_state=42), 
            "params" : {
                'max_depth' : [5,10, None], 
                'min_samples_split' : [2,5]
            }
        }, 
        "NaiveBayes" : {
            "model" : BernoulliNB(), 
            "params" : {
                'alpha' : [0.1, 0.5, 1.0]
            }
        }, 
        "LogisticRegresion" : {
            "model" : LogisticRegression(random_state=42, max_iter=1000), 
            "params" : {
                'C' : [0.1, 1, 10], 
                'solver' : ['liblinear', 'lbfgs']
            }
        }
    } 
    print("Start Training")
    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}")
        with mlflow.start_run(run_name=f"Tuning_{model_name}"):
            grid = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=3, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_params = grid.best_params_ 

            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"Best Params : {best_params}")
            print(f"Accuracy : {acc}")

            mlflow.log_params(best_params)
            mlflow.log_param("algorithm", model_name)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1) 

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')

            cm_file = f"cm_{model_name}.png"
            plt.savefig(cm_file)
            plt.close()
            

            mlflow.log_artifact(cm_file)

            if os.path.exists(cm_file): os.remove(cm_file) 

            if hasattr(best_model, "predict_proba"):
                try:
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr) 

                    plt.figure(figsize=(6,5))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc='lower right') 

                    roc_file = f"roc_{model_name}.png"
                    plt.savefig(roc_file)
                    plt.close()
                    mlflow.log_artifact(roc_file)
                    if os.path.exists(roc_file) : os.remove(roc_file) 
                    mlflow.log_metric("auc_score", roc_auc) 
                except Exception as e:
                    print(f"Skipping ROC for {model_name}: {e}") 
            
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    best_model, X_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5) 
                ) 

                train_scores_mean = np.mean(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)

                plt.figure(figsize=(6,5))
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
                plt.title(f"Learning Curve - {model_name}")
                plt.xlabel("Training Examples")
                plt.ylabel("Score")
                plt.legend(loc="best")
                plt.grid() 

                lc_file = f"learning_curve_{model_name}.png" 
                plt.savefig(lc_file)
                plt.close()
                mlflow.log_artifact(lc_file)
                if os.path.exists(lc_file): os.remove(lc_file)
            except Exception as e:
                print(f" Skipping Learning Curve {model_name}: {e}") 

            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="model",
                registered_model_name=f"Model_{model_name}"
            )
    print("Done")

if __name__ == "__main__":
    train()
    