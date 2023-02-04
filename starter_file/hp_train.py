from sklearn.svm import SVC
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def prepare_data(data):
    input_cols = np.arange(1,14,1)
    output_cols = [14]
    x = data[data.columns[input_cols]]
    y = data[data.columns[output_cols]]
    return x, y

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Regularization Parameter")
    parser.add_argument('--kernel', type=str, default='rbf', help="Kernel Type")
    parser.add_argument('--degree', type=int, default=3, help="Degree of polynomial function")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Parameter:", np.float(args.C))
    run.log("Kernel:", str(args.kernel))
    run.log("Degree:", np.int(args.degree))

    model = SVC(C=args.C, kernel=args.kernel, degree=args.degree).fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    run.log("Accuracy", np.float(acc))

if __name__ == '__main__':
    #Had to relocate my user code here otherwise Job would not start

    ds = pd.read_csv('salary_cleaned.csv')
    
    x, y = prepare_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)

    main()
