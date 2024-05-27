import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
if __name__ == '__main__':
    dataframes = pd.read_excel("/gpfs/home/P02114015/faceReconition/HOG/traits.xlsx")
    features = dataframes.iloc[:,0:-1]
    labels = dataframes.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=213)

    model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True) 
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print(y_pred)
    ok = 0
    for i in range(y_pred.shape[0]):
        if(y_pred[i] == y_test.iloc[i].data):
            ok += 1
    print(ok,y_pred.shape[0])
    print(f'Accuracy: {ok*100.0 / y_pred.shape[0]:.2f}%')
    joblib.dump(model,"/gpfs/home/P02114015/faceReconition/FaceDetaction/HOG_detaction.pkl")
    # tmp = np.load("./traits.npy")
    # print(model.predict(tmp.reshape(1,-1)[:,:-1]))
    