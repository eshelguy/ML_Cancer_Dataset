from algorithms import algo_knn, algo_svm, algo_pca, algo_logistic_regression
from utils import plot_comparison


def choose_algorithm():
    print("בחר אלגוריתם:")
    print("1. KNN")
    print("2. SVM")
    print("3. PCA")
    print("4. Logistic Regression")

    choice = input("הכנס את מספר האלגוריתם הרצוי: ")

    return choice


if __name__ == '__main__':
    # רשימות לשמירת התוצאות של כל האלגוריתמים
    knn_scores = None
    svm_scores = None
    pca_scores = None
    logistic_scores = None

    # קלט מהמשתמש לבחירת אלגוריתם
    while True:
        choice = choose_algorithm()

        if choice == '1':
            print("בחרת ב-KNN")
            n_neighbors = int(input("הכנס את מספר השכנים הרצוי ב-KNN (מספר שלם): "))
            knn_scores = algo_knn(10,True,n_neighbors)
        elif choice == '2':
            print("בחרת ב-SVM")
            svm_scores = algo_svm(10,True)
        elif choice == '3':
            print("בחרת ב-PCA")
            n_components = int(input("הכנס את מספר הממדים הרצוי ב-PCA (מספר שלם): "))
            pca_scores = algo_pca(10,True , n_components)
        elif choice == '4':
            print("בחרת ב-Logistic Regression")
            logistic_scores = algo_logistic_regression(10,True)
        else:
            print("בחירה לא תקפה. נסה שוב.")

        # שואל אם המשתמש רוצה להוסיף עוד אלגוריתם או לסיים
        another = input("האם אתה רוצה לבחור אלגוריתם נוסף? (y/n): ")
        if another.lower() != 'y':
            break


    # הרצת כל האלגוריתמים גם אם לא נבחרו
    if not knn_scores:
        knn_scores = algo_knn()
    if not svm_scores:
        svm_scores = algo_svm()
    if not pca_scores:
        pca_scores = algo_pca()
    if not logistic_scores:
        logistic_scores = algo_logistic_regression()

    # הצגת תוצאות בגרף לאחר סיום הבחירות
    # אם יש אלגוריתמים שנבחרו, הצג את הגרף
    plot_comparison(
        knn_scores if knn_scores else [],
        svm_scores if svm_scores else [],
        pca_scores if pca_scores else [],
        logistic_scores if logistic_scores else []
    )