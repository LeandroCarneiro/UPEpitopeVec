import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def MultiModelPlotROC(models, y_test):
    plt.figure()

    for model in models:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, model.prediction)

        # Calculate AUC
        auc = roc_auc_score(y_test, model.prediction)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{model.name} ROC (area = {auc:.2f})')

    # Custom settings for the plot
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
