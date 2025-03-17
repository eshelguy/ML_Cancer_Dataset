import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report


def get_features_and_labels():
    path = 'data/Cancer_Data.csv'
    # טעינת הcsv
    data = pd.read_csv(path)
    #מחיקת עמודות לא בשימוש
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    #הפיכת עמודה diagnosis מB שפיר לm סרטני ל 0,1 באמצעות LabelEncoder
    label_encoder = LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])  #B -> 0 , M -> 1,
    #לתוך X נכנס המידע ללא הקלסיפקציה
    X = data.drop('diagnosis', axis=1)
    # בY יושב הקלסיפקציה
    Y = data['diagnosis']
    return X, Y


def plot_result(y_test, y_pred, score_train, score_test, algo_name):
    #תוצאות האלגוריתם
    print(f'{algo_name} - ממוצע דיוק אימון:', score_train)
    print(f'{algo_name} - ממוצע דיוק בדיקה:', score_test)

    #ציור מטריצת החיזוי
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

#ציור השוואה בין כל האלגוריתמים
def plot_comparison(knn_scores, svm_scores, logistic_scores, random_forest):
    algorithms = ['KNN', 'SVM', 'Logistic Regression', 'Random Forest']
    train_scores = [score[0] for score in [knn_scores, svm_scores, logistic_scores, random_forest]]
    test_scores = [score[1] for score in [knn_scores, svm_scores, logistic_scores, random_forest]]

    x = range(len(algorithms))

    plt.bar(x, train_scores, width=0.4, label='Train Accuracy', align='center', alpha=0.7)
    plt.bar(x, test_scores, width=0.4, label='Test Accuracy', align='edge', alpha=0.7)

    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Algorithms')
    plt.xticks(x, algorithms)
    plt.legend()
    plt.show()

def plot_comparison_pca(knn_scores, svm_scores, logistic_scores, random_forest):
    algorithms = ['KNN+PCA', 'SVM+PCA', 'LR+PCA', 'RF+PCA']
    train_scores = [score[0] for score in [knn_scores, svm_scores, logistic_scores, random_forest]]
    test_scores = [score[1] for score in [knn_scores, svm_scores, logistic_scores, random_forest]]

    x = range(len(algorithms))

    plt.bar(x, train_scores, width=0.4, label='Train Accuracy', align='center', alpha=0.7)
    plt.bar(x, test_scores, width=0.4, label='Test Accuracy', align='edge', alpha=0.7)

    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Algorithms + PCA')
    plt.xticks(x, algorithms)
    plt.legend()
    plt.show()
