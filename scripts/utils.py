# Import the necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_churn_confusion_matrix(y_test, y_pred, feature, model_name):
    """
    Plots the confusion matrix for a classification model.

    Parameters:
    - y_test: Ground truth target values.
    - y_pred: Predicted target values.
    - feature: Indicates the feature treated (numerical or combination)
    - model_name: Name of the model (used for the plot title).
    """
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"accuracy : {accuracy:.4f}, f1_score : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}({feature})")
    file_name = f"../outputs/confmat_{feature}_{model_name}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()
