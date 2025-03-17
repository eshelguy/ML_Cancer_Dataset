from algorithms import algo_knn, algo_svm, algo_random_forest, algo_logistic_regression
from utils import plot_comparison, plot_comparison_pca


def choose_algorithm():
    print("בחר אלגוריתם:")
    print("1. KNN")
    print("2. SVM")
    print("3. Logistic Regression")
    print("4. Random Forest")

    choice = input("הכנס את מספר האלגוריתם הרצוי: ")

    return choice


if __name__ == '__main__':
    # רשימות לשמירת התוצאות של כל האלגוריתמים
    knn_scores = None
    svm_scores = None
    logistic_scores = None
    random_forest_scores = None

    knn_scores_pca = None
    svm_scores_pca = None
    logistic_scores_pca = None
    random_forest_scores_pca = None

    # קלט מהמשתמש לבחירת אלגוריתם
    while True:
        choice = choose_algorithm()
        apply_pca = input("האם להפעיל PCA? (y/n): ").lower() == 'y'
        n_components = 2  # ברירת מחדל ל-PCA

        if apply_pca:
            n_components = int(input("הכנס את מספר הממדים הרצוי ב-PCA (מספר שלם): "))

        if choice == '1':
            print("בחרת ב-KNN")
            n_neighbors = int(input("הכנס את מספר השכנים הרצוי ב-KNN (מספר שלם): "))
            if apply_pca:
                knn_scores_pca = algo_knn(10, True, n_neighbors, apply_pca, n_components)
            else:
                knn_scores = algo_knn(10, True, n_neighbors, apply_pca, n_components)
        elif choice == '2':
            print("בחרת ב-SVM")
            if apply_pca:
                svm_scores_pca = algo_svm(10, True, apply_pca, n_components)
            else:
                svm_scores = algo_svm(10, True, apply_pca, n_components)
        elif choice == '3':
            print("בחרת ב-Logistic Regression")
            if apply_pca:
                logistic_scores_pca = algo_logistic_regression(10, True, apply_pca, n_components)
            else:
                logistic_scores = algo_logistic_regression(10, True, apply_pca, n_components)
        elif choice == '4':
            print("בחרת ב-Random Forest")
            n_estimators = int(input("הכנס את מספר העצים ב-Random Forest (מספר שלם): "))
            if apply_pca:
                random_forest_scores_pca = algo_random_forest(10, True, n_estimators, apply_pca, n_components)
            else:
                random_forest_scores = algo_random_forest(10, True, n_estimators, apply_pca, n_components)
        else:
            print("בחירה לא תקפה. נסה שוב.")
            continue

        another = input("האם אתה רוצה לבחור אלגוריתם נוסף? (y/n): ")
        if another.lower() != 'y':
            break


    # הרצת כל האלגוריתמים גם אם לא נבחרו
    if not knn_scores:
        knn_scores = algo_knn(10, True, 5, False, 2)
    if not svm_scores:
        svm_scores = algo_svm(10, True, False, 2)
    if not logistic_scores:
        logistic_scores = algo_logistic_regression(10, True, False, 2)
    if not random_forest_scores:
        random_forest_scores = algo_random_forest(10, True, 100, False, 2)

    if not knn_scores_pca:
        knn_scores_pca = algo_knn(10, True, 5, True, 2)
    if not svm_scores_pca:
        svm_scores_pca = algo_svm(10, True, True, 2)
    if not logistic_scores_pca:
        logistic_scores_pca = algo_logistic_regression(10, True, True, 2)
    if not random_forest_scores_pca:
        random_forest_scores_pca = algo_random_forest(10, True, 100, True, 2)

    # הצגת תוצאות בגרף לאחר סיום הבחירות
    # אם יש אלגוריתמים שנבחרו, הצג את הגרף
    plot_comparison(
        knn_scores if knn_scores else [],
        svm_scores if svm_scores else [],
        logistic_scores if logistic_scores else [],
        random_forest_scores if random_forest_scores else []
    )

    plot_comparison_pca(
        knn_scores_pca if knn_scores_pca else [],
        svm_scores_pca if svm_scores_pca else [],
        logistic_scores_pca if logistic_scores_pca else [],
        random_forest_scores_pca if random_forest_scores_pca else []
    )
