
# Customer Churn Prediction App
This project is a web-based application for predicting customer churn.
It uses a deep learning model trained on customer data to provide insights into whether a given customer is likely to churn or not. 
The model takes various inputs such as contract type, monthly charges, total charges,tenure,gender and paperless billing to make predictions.

## Authors

- [@hilaryookrah]( https://github.com/hilaryookrah/90832025_Churning_Customer)


## Appendix

## Features
- **User-Friendly Interface**: The app provides a simple and intuitive interface for users to input customer data and receive predictions.

- **Deep Learning Model**: The predictive model is based on a deep learning architecture, making accurate predictions on customer churn.

- **Scalability**: The application is designed to be scalable, allowing for easy integration with additional features and improvements.


## Dependencies
Streamlit
Pandas
NumPy
TensorFlow
Scikit-Learn
Keras
## File Structure
app.py: Main Streamlit application script.


prediction_model.h5: Trained deep learning model.


standard_scaler.pkl: Pickle file for the StandardScaler used in preprocessing.
## Deployment

To deploy this project run

```bash
 
```



## Data Preprocessing

- The dataset is loaded from a CSV file (`datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- Irrelevant columns like 'customerID' are dropped.
- Data types are checked and converted, and missing values are handled appropriately.

## 🚀 About Me
Hilary Achiaa Osei-Okrah

Student at Ashesi University

