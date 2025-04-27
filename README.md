# Fraud-Detection-

Fraud Detection Project
This project addresses the fraud detection problem by building and evaluating models to identify fraudulent transactions in a financial dataset. Fraud detection is a critical task in finance and e-commerce: it involves distinguishing between legitimate and fraudulent transactions, often in the presence of heavily imbalanced classes. The goal of this project is to explore data-driven techniques that can flag suspicious activities and prevent financial loss. It uses machine learning and deep learning approaches to classify transactions as fraudulent or non-fraudulent, leveraging the sequential and numerical features available in the data.


Dataset
The dataset used contains 10,033 transaction records (examples) with features such as transaction time step, type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT), transaction amount, and account balances before and after the transaction. Key columns include step, type, amount, nameOrig (originator account), oldbalanceOrg, newbalanceOrig, nameDest (destination account), oldbalanceDest, newbalanceDest, and the target label isFraud. There is also an isFlaggedFraud column (which in this data was constant at 0) indicating whether a transaction was automatically flagged. 
The data had no missing values and was moderately sized, enabling quick experimentation. (This data is sourced from a synthetic online payments fraud dataset with customer and merchant accounts.) Because the number of fraud cases is very small compared to non-fraudulent transactions, the classes are highly imbalanced, which poses a challenge for model training.


Methodology
The approach involves the following major steps and techniques:
Data Preprocessing: We first checked for missing values (none were found) and applied encoding to categorical fields. Specifically, the type, nameOrig, and nameDest columns were label-encoded into numeric form. All features were then scaled using a standard scaler to normalize their ranges. We split the data into training and testing sets with an 80/20 stratified split, ensuring the class imbalance is reflected in both sets. In some experiments, class weights were used to give more importance to the minority (fraud) class during training.

Modeling:
Several models were implemented and compared:
A Temporal Convolutional Network (TCN): This is a type of convolutional neural network designed for sequential data. The model consists of one TCN layer (with 64 filters) followed by a dense output neuron. It was trained with binary cross-entropy loss and measured on accuracy. TCNs can capture temporal patterns in the transaction sequence without using recurrent units.

A CNN–SVM Hybrid: We built a 1D Convolutional Neural Network (Conv1D) as a feature extractor (with Conv1D, MaxPooling, and dense layers) to process the scaled features. The flattened output features from the CNN were then used to train an SVM (Support Vector Machine) classifier (with an RBF kernel). This hybrid approach leverages CNNs for automatic feature learning and SVMs for classification.

An LSTM Network: A standard Long Short-Term Memory (LSTM) recurrent neural network (with one LSTM layer of 64 units and dropout) was also applied to the sequential data. The input features were reshaped appropriately for the LSTM. This model is capable of capturing long-term dependencies in the transaction sequence.

XGBoost Classifier: 
A gradient boosting tree ensemble (XGBoost) was trained on the scaled features. XGBoost is a powerful classifier that often performs well on tabular data, including for fraud detection. It was included as a strong baseline and ultimately gave the best balanced performance.

Evaluation:
Each model was evaluated on the test set using accuracy, and additionally we performed 5-fold stratified cross-validation to measure metrics such as precision, recall, F1-score, and ROC AUC. This helps account for the class imbalance. The confusion matrix and ROC curves were also examined for further insight. In general, high overall accuracy was achieved by predicting mostly non-fraud cases (due to imbalance), so the precision and recall of the fraud class are especially important.
Results

All models achieved very high raw accuracy (around 99% or higher) because the dataset is dominated by non-fraud cases. However, examining precision and recall reveals their effectiveness on fraud detection:

XGBoost: 
The XGBoost model performed best in cross-validation, with an average accuracy around 99.72%, precision ~96.18%, recall ~61.87%, F1-score ~74.76%, and ROC AUC ~99.85%. This indicates it correctly identifies a high fraction of fraud cases (over 60% recall) while maintaining very few false positives (high precision). XGBoost was able to exploit the feature patterns in the data effectively.

CNN–SVM: 
The CNN feature extractor combined with SVM achieved test accuracy around 99.50%. In cross-validation, it showed accuracy ~99.38%, precision ~80.0%, recall ~10.4%, and ROC AUC around 96.7%. The high precision suggests it rarely flags non-fraud as fraud, but the low recall indicates it detects only a small portion of actual fraud cases. This model learned strong features but still mostly predicted the majority class.

TCN (Temporal CNN):
The TCN model reached about 99.50% test accuracy as well. Its cross-validated performance was around 90.23% accuracy (much lower than others) with precision ~5.8%, recall ~92.3%, and ROC AUC ~97.59%. The very low precision means it had many false alarms, while the high recall shows it caught most fraud cases. This behavior suggests the TCN tended to predict fraud more often to avoid missing them, at the cost of many false positives.

LSTM: 
The LSTM achieved about 99.35% test accuracy. However, in cross-validation it effectively had 0% precision and recall on the fraud class (accuracy ~99.31%, ROC AUC ~91.38%). This means the LSTM model almost always predicted “non-fraud” in each fold (likely due to the imbalance and perhaps insufficient epochs), so it missed all actual fraud cases during cross-validation. In summary, the LSTM and some other models defaulted to predicting the majority class.

These results highlight the class imbalance issue: models can achieve >99% accuracy by predicting no fraud, but such models are useless for detection. The XGBoost model struck a better balance (high precision and moderate recall). The deep learning models showed that careful class weighting or further tuning would be needed to improve recall. Overall, the project demonstrates that ensemble tree methods like XGBoost can be very effective on this fraud dataset, while deep neural approaches require extra handling of imbalance.

Conclusion

This project explored multiple approaches to fraud detection on a transactional dataset. By preprocessing the data, encoding categorical fields, and scaling features, we prepared the data for modeling. We implemented different sequential and non-sequential models (TCN, CNN+SVM, LSTM, and XGBoost) and evaluated them using stratified cross-validation. The key takeaway is that while neural models can learn complex patterns, a well-tuned XGBoost classifier achieved the best trade-off between catching frauds and avoiding false alarms. All code and results are available in this repository. We welcome feedback, issues, and contributions from other developers and data scientists interested in fraud analytics. Feel free to fork the project, suggest improvements, or collaborate on extending this work!
