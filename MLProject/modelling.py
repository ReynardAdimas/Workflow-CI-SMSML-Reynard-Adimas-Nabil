import pandas as pd
import os 
import mlflow 
import mlflow.sklearn
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump


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
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI Experiment")
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
        with mlflow.start_run() as run:
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

            # model_file = f"model_{model_name}.joblib"
            # dump(best_model, model_file)
             
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="model",
                registered_model_name=f"Model_{model_name}"
            )

            mlflow.log_artifact(cm_file)
            # mlflow.log_artifact(model_file)

            if os.path.exists(cm_file): os.remove(cm_file)
            #if os.path.exists(model_file) : os.remove(model_file) 
            with open("last_run_id.txt", "w") as f:
                f.write(run.info.run_id)
    print("Done")

if __name__ == "__main__":
    train()
    