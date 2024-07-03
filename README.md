# Parkinsons-Disease-Detection


## SVM-Based Parkinson's Disease Detection

This project involves using a Support Vector Machine (SVM) to detect Parkinson's disease based on a dataset obtained from Kaggle. The model achieves an accuracy of 87.17%.

### Dataset

The dataset used for this project is publicly available on Kaggle and can be found [here](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set). The dataset contains a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease.

### Files

- `parkinsons.data`: The main dataset file containing the features and labels.
- `README.md`: This file, containing an overview of the project.

### Features

The dataset includes the following features:

1. **name** - ASCII subject name and recording number
2. **MDVP:Fo(Hz)** - Average vocal fundamental frequency
3. **MDVP:Fhi(Hz)** - Maximum vocal fundamental frequency
4. **MDVP:Flo(Hz)** - Minimum vocal fundamental frequency
5. **MDVP:Jitter(%), MDVP:Jitter(Abs), RAP, PPQ, DDP** - Several measures of variation in fundamental frequency
6. **MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA** - Several measures of variation in amplitude
7. **NHR, HNR** - Two measures of ratio of noise to tonal components in the voice
8. **status** - Health status of the subject (one) - Parkinson's, (zero) - healthy
9. **RPDE, D2** - Two nonlinear dynamical complexity measures
10. **DFA** - Signal fractal scaling exponent
11. **spread1, spread2, PPE** - Three nonlinear measures of fundamental frequency variation

### Model

The model used is a Support Vector Machine (SVM). The SVM is trained to classify individuals as having Parkinson's disease or being healthy based on the given features.

### Performance

The SVM model achieves an accuracy of 87.17% on the dataset.

### Dependencies

To run this project, the following dependencies are required:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

### Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```

3. Load and preprocess the dataset:
   ```python
   import pandas as pd

   df = pd.read_csv('parkinsons.data')
   ```

4. Train the SVM model:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   from sklearn.metrics import accuracy_score

   X = df.drop(['name', 'status'], axis=1)
   y = df['status']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)

   svm = SVC(kernel='linear')
   svm.fit(X_train, y_train)
   y_pred = svm.predict(X_test)

   accuracy = accuracy_score(y_test, y_pred)
   print(f"Model Accuracy: {accuracy * 100:.2f}%")
   ```

### Conclusion

This project demonstrates the use of an SVM for detecting Parkinson's disease using biomedical voice measurements. The model achieves a notable accuracy of 87.17%, highlighting the potential of machine learning in medical diagnostics.
