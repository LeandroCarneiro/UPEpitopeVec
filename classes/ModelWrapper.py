from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelWrapper:
    def __init__(self, name, instance, prediction, scores, accuracy):
        self.name = name
        self.instance = instance
        self.prediction = prediction
        self.scores = scores
        self.accuracy = accuracy

    def ConfusionMatrix(self, expected, show=False):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            expected, (self.prediction > 0.5).astype(int))
        disp.figure_.suptitle("Confusion Matrix")

        if (show):
            plt.show()

    def Report(self, expected):
        print(classification_report(expected, self.prediction, digits=4))
        print('-' * 50)
        print(f'\n')

    def Histogram(self, expected, show=False):
       # Initialize data list
        data = []

        kappa_indices = np.array(cohen_kappa_score(
            expected, (self.prediction > 0.5).astype(int)))

        # Append dictionaries to the list
        data.append({"value": kappa_indices, "name": "Kappa"})
        data.append({"value": self.accuracy, "name": "Accuracy"})

        # Separate names and values for plotting
        names = [item["name"] for item in data]
        values = [item["value"] for item in data]

        # Define colors for each bar (example colors)
        colors = ['red', 'blue']  # Adjust as needed

        # Create figure and axis objects
        fig, ax = plt.subplots()

        # Plot histogram/bar chart
        ax.bar(names, values, color=colors)

        # Set labels and title
        ax.set_title('Comparison of Kappa and Accuracy')
        ax.set_xlabel('Measurements')
        ax.set_ylabel('Value')

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
