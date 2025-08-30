# Metrics

---

![Metrics Summary](https://images.prismic.io/encord/edfa849b-03fb-43d2-aba5-1f53a8884e6f_image5.png?auto=compress,format)

## 1. Precision
- **Definition:**  
  Precision is the proportion of positive predictions that are actually correct. It answers the question: *Of all the instances predicted as positive, how many were actually positive?*

- **Example:**  
  In a spam email classification, if the model identifies 100 emails as spam and 80 of them are actually spam, the precision is 0.8 or 80%.

- **Notes:**  
  - Formula:  
    \( \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} \)
  - High precision means low false positives.

---

## 2. Recall
- **Definition:**  
  Recall (or Sensitivity) is the proportion of actual positives that were correctly identified. It answers: *Of all the actual true instances, how many are predicted as positive?*

- **Example:**  
  In a medical test for a disease, if 100 people have the disease and the model identifies 90 of them, recall would be 0.9 or 90%.

- **Notes:**  
  - Formula:  
    \( \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} \)
  - High recall means low false negatives.

---

## 3. Accuracy
- **Definition:**  
  Accuracy is the proportion of correct predictions (both true positives and true negatives) among all predictions. It answers: *How many predictions were correct overall?*

- **Example:**  
  If the model correctly predicts 90 out of 100 instances, the accuracy is 90%.

- **Notes:**  
  - Formula:  
    \( \text{Accuracy} = \frac{\text{True Positives + True Negatives}}{\text{Total Instances}} \)
  - Accuracy can be misleading in imbalanced datasets.

---

## 4. Specificity
- **Definition:**  
  Specificity is the proportion of actual negatives that were correctly identified. It answers: *Of all the actual negatives, how many were correctly predicted as negative?*

- **Example:**  
  In a fraud detection system, if 100 transactions are non-fraudulent and the model identifies 80 of them as non-fraudulent (true negatives), the specificity would be 0.8 or 80%.

- **Notes:**  
  - Formula:  
    \( \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives + False Positives}} \)
  - High specificity means low false positives.

---

## 5. Confusion Matrix
- **Definition:**  
  A confusion matrix is a table used to evaluate the performance of a classification model. It shows the actual vs predicted classifications across four categories: true positives, false positives, true negatives, and false negatives.

- **Example:**  
  For a binary classification, the matrix might look like:
  
  |                 | Predicted Positive | Predicted Negative |
  |-----------------|--------------------|--------------------|
  | Actual Positive | True Positive (TP)  | False Negative (FN)|
  | Actual Negative | False Positive (FP) | True Negative (TN) |

- **Notes:**  
  - Provides a clear breakdown of how many errors the model made.
  - Helps derive other metrics like Precision, Recall, etc.

---

## 6. F1 Score
- **Definition:**  
  F1 score is the harmonic mean of precision and recall, used to balance the trade-off between the two. It is particularly useful when dealing with imbalanced datasets.

- **Example:**  
  If precision = 0.8 and recall = 0.6, then:  
  \( F1 = \frac{2 \times (0.8 \times 0.6)}{0.8 + 0.6} = 0.685 \)

- **Notes:**  
  - Formula:  
    \( F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \)
  - Ranges between 0 (worst) and 1 (best).

---

## 7. ROC Curve
- **Definition:**  
  The ROC (Receiver Operating Characteristic) curve is a graphical representation of the trade-off between the true positive rate (recall) and the false positive rate across different threshold values.

- **Example:**  
  The ROC curve plots the true positive rate (y-axis) against the false positive rate (x-axis). A model with good performance will have a curve closer to the top left corner.

![ROC Curve Example](https://i0.wp.com/sefiks.com/wp-content/uploads/2020/12/roc-curve-original.png?fit=726%2C576&ssl=1)

- **Notes:**  
  - The area under the ROC curve (AUC) can be used to quantify model performance.
  - Helps visualize how well a model distinguishes between classes.

---

## 8. AUC (Area Under the Curve)
- **Definition:**  
  AUC represents the area under the ROC curve. It provides a single scalar value to evaluate the performance of a model, ranging from 0 to 1. A higher AUC indicates better model performance.

- **Example:**  
  If a model has an AUC of 0.9, it means the model has a 90% chance of correctly distinguishing between a positive and a negative instance.

- **Notes:**  
  - AUC = 0.5 indicates a model with no discrimination ability (similar to random guessing).
  - AUC = 1 indicates a perfect model.

---