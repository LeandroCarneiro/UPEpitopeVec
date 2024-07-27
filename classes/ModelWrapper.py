from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelWrapper:
    def __init__(self, name, instance, prediction, scores):
        self.name = name
        self.instance = instance
        self.prediction = prediction
        self.scores = scores

    def ConfusionMatrix(self, expected, show=False):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            expected, (self.prediction > 0.5).astype(int))
        disp.figure_.suptitle("Confusion Matrix")

        if (show):
            plt.show()

    def Report(self, expected, predicted):
        print(classification_report(expected, predicted, digits=4))
        print('-' * 50)
        print(f'\n')

    def KappaHistogram(self, expected, predicted, show=False):
        kappa_indices = np.array(cohen_kappa_score(expected, predicted))

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(kappa_indices, bins='auto', density=True,
                 alpha=0.7, color='blue', edgecolor='black')

        plt.title('Histogram of Kappa Indices')
        plt.xlabel('Kappa Index Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        if (show):
            plt.show()

    def ReportAcuracy(self, expected):
        y_pred = (self.prediction > 0.5).astype(int)
        accuracy = accuracy_score(expected, y_pred)
        print(f'Acuracy: {accuracy:.4f}')
        print(f'\n')

    def ReportCrossValidation(self):
        print(f'Cross Validation: {
              np.mean(self.scores):.4f} (+/- {np.std(self.scores):.4f})')

    def ReportROC(self, expected, show=False):
        plt.figure()
        fpr, tpr, _ = roc_curve(expected, self.prediction)
        auc = roc_auc_score(expected, self.prediction)
        plt.plot(fpr, tpr, label=f'{self.name} ROC (area = {auc:.2f})')

        # Custom settings for the plot
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if (show):
            plt.show()
