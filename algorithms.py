from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import get_features_and_labels, plot_result

#פונקציה שמציירת ועושה ממוצע לפי כמות הiterations
def evaluate_algorithm(algo, X_train, X_test, y_train, y_test, algo_name, iterations=100, print_results=True):
    sum_train = 0
    sum_test = 0

    for _ in range(iterations):
        #אימון המודל
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
        #תוצאות על האימון
        score_train = accuracy_score(y_train, algo.predict(X_train))
        #תוצאות על הבדיקה
        score_test = accuracy_score(y_test, y_pred)

        sum_train += score_train
        sum_test += score_test
    #ממוצע ככמות האיטרציות שנשלחו לפונקציה
    avg_train = sum_train / iterations
    avg_test = sum_test / iterations
    #הדפסה של מטריצת החיזוי
    if print_results:
        plot_result(y_test, y_pred, avg_train, avg_test, algo_name)

    return avg_train, avg_test


def algo_knn(num_iterations=100, print_results=False ,n_neighbors = 5, apply_pca=False, n_components=2):

    #מביא את הנתונים מהcsv מהפונקציה שבניתי
    X, Y = get_features_and_labels()
    # מחלקת את הנתונים לסט אימון וסט בדיקה. 80% לאימון ו-20% לבדיקה. ומערבבת את הנתונים באמצעות shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    scaler = StandardScaler()
    #מנרמלים את הנתונים באמצעות המחלקה scaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    #אובייקט האלגוריתם שנבחר כאן נבחר KNN כאשר n_neighbors זה מספר השכנים ויכול להשתנות בשליחה לפונקציה הדיפולט 5
    algo = KNeighborsClassifier(n_neighbors=n_neighbors)
    return evaluate_algorithm(algo, X_train, X_test, y_train, y_test, 'KNeighbors',num_iterations ,print_results)


def algo_svm(num_iterations=100, print_results=False, apply_pca=False, n_components=2):
    X, Y = get_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    algo = SVC(kernel='linear', random_state=42)
    return evaluate_algorithm(algo, X_train, X_test, y_train, y_test, 'SVM',num_iterations ,print_results)


def algo_logistic_regression(num_iterations=100, print_results=False, apply_pca=False, n_components=2):
    X, Y = get_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    algo = LogisticRegression(solver='liblinear', random_state=0)
    return evaluate_algorithm(algo, X_train, X_test, y_train, y_test, 'Logistic Regression',num_iterations ,print_results)


def algo_random_forest(num_iterations=100, print_results=False, n_estimators=100, apply_pca=False, n_components=2):
    X, Y = get_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    algo = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    return evaluate_algorithm(algo, X_train, X_test, y_train, y_test, 'Random Forest', num_iterations, print_results)
