import numpy as np
import matplotlib.pyplot as plt

# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
def calculate_accuracy(y_pred, labels):
  y_pred_label = np.argmax(y_pred, axis=1)
  true_positives = np.sum(labels == y_pred_label)
  accuracy = 100 * true_positives / len(y_pred)
  print(f"Overall accuracy = {accuracy:2.1f}")
  return accuracy    

# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
# classes are the model's classes
def calculate_all_accuracies(y_pred, labels, classes):
  n_classes = len(classes)

  # Initialize array of accuracies
  accuracies = np.zeros(n_classes)

  # Loop on classes
  for class_item in range(n_classes):
    true_positives = 0
    # Loop on all predictions
    for i in range(len(y_pred)):
      # Check if it matches the class that we are working on
      if( labels[i] == class_item ):
        # Get prediction label
        y_pred_label = np.argmax(y_pred[i,:])
        # Check if the prediction is correct
        if( labels[i] == y_pred_label ):
          true_positives += 1

    accuracies[class_item] = 100 * true_positives / np.sum(labels == class_item)
    print(f"Accuracy = {accuracies[class_item]:2.1f} ({classes[class_item]})")
    
  return accuracies

# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
# classes are the model's classes
# name is the model's name
def calculate_auc(y_pred, labels, classes,name):
  n_classes = len(classes)
  
  # thresholds, linear range, may need improvements for better precision 
  thresholds = np.arange(0.0, 1.01, .01)
  # false positive rate
  fpr = np.zeros([n_classes,len(thresholds)])
  # true positive rate
  tpr = np.zeros([n_classes,len(thresholds)])
  # area under curve
  roc_auc = np.zeros(n_classes)

  # get number of positive and negative examples in the dataset
  for class_item in range(n_classes):
    # Sum of all true positive answers
    all_positives = sum(labels == class_item)
    # Sum of all true negative answers
    all_negatives = len(labels) - all_positives

    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh_item in range(1,len(thresholds)):
      thresh = thresholds[thresh_item]
      false_positives = 0
      true_positives = 0
      for i in range(len(y_pred)):
        # Check prediction for this threshold
        if (y_pred[i, class_item] > thresh):
          if labels[i] == class_item:
            true_positives += 1
          else:
            false_positives += 1
      fpr[class_item,thresh_item] = false_positives/float(all_negatives)
      tpr[class_item,thresh_item] = true_positives/float(all_positives)

    # Force boundary condition
    fpr[class_item,0] = 1
    tpr[class_item,0] = 1

    # calculate area under curve, trapezoid integration
    for thresh_item in range(len(thresholds)-1):
      roc_auc[class_item] += .5*(tpr[class_item,thresh_item]+tpr[class_item,thresh_item+1])*(fpr[class_item,thresh_item]-fpr[class_item,thresh_item+1]);

  # results
  roc_auc_avg = np.mean(roc_auc)
  print(f"Simplified average roc_auc = {roc_auc_avg:.3f}")               

  plt.figure()
  for class_item in range(n_classes):
    plt.plot(fpr[class_item,:], tpr[class_item,:], label=f"auc: {roc_auc[class_item]:0.3f} ({classes[class_item]})")
  plt.xlim([0.0, 0.1])
  plt.ylim([0.5, 1.0])
  plt.legend(loc="lower right")
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC: '+name)
  plt.grid(which='major')
  plt.show(block=False)
  
  return roc_auc
