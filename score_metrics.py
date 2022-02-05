import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def multiclass_accuracy(preds, truths):
    """
    Computes the multiclass accuracy

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    acc = np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    return acc

def mosei_scores(preds, truths, message=''):
    print(message)
    test_preds = preds.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_7 = np.clip(test_truth, a_min=-3., a_max=3.)

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)

    mult_acc7 = multiclass_accuracy(test_preds_7, test_truth_7)
    print("Multi Accuracy 7: ", np.round(mult_acc7, 5))

    binary_preds = (test_preds >= 0)
    binary_truth = (test_truth >= 0)
    f_score = f1_score(binary_truth, binary_preds, average='weighted')
    binary_accuracy = accuracy_score(binary_truth, binary_preds)
    print("Accuracy: ", np.round(binary_accuracy, 5))
    print("F1 score: ", np.round(f_score, 5))

    print("-" * 50)
    return f_score
