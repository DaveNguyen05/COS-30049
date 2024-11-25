# Melbourne Property Price Prediction Project : Fantastic Realtors

## Overview

This project aims to predict property prices in Melbourne using machine learning models. The dataset contains various features such as the number of rooms, property size, location, and more. We explored different machine learning algorithms including regression and clustering models to analyze the dataset and generate predictions.

The primary objective is to build a robust regression model that accurately predicts the price of a property based on its features. The project includes data preprocessing, model training, and evaluation using Random Forest Regressor, Gradient Boosting Regressor, and K-Means Clustering.

## Dataset

The dataset used in this project is `Property Sales of Melbourne City.csv`, containing historical sales data of properties in Melbourne. The following features were used for model building:

- **Suburb**: The suburb where the property is located.
- **Rooms**: The number of rooms in the property.
- **Type**: Type of property (house, unit, etc.).
- **Price**: Sale price of the property (target variable).
- **Distance**: Distance from the central business district.
- **Bathroom**: Number of bathrooms.
- **Car**: Number of car spots.
- **Landsize**: Size of the land in square meters.
- **BuildingArea**: Size of the building in square meters.
- **YearBuilt**: Year the property was built.
- **Lattitude**: Latitude of the property location.
- **Longtitude**: Longitude of the property location.
- **Regionname**: The region where the property is located.

## Project Structure

- **Data Preprocessing**: The dataset is cleaned and preprocessed by handling missing values, applying one-hot encoding to categorical variables, and normalizing numerical data.
- **Model Training and Evaluation**:
  - **Random Forest Regressor**: A tree-based ensemble method that averages the predictions of multiple decision trees to improve accuracy.
  - **Gradient Boosting Regressor**: An iterative ensemble model that corrects errors made by previous models, improving predictive power.
  - **K-Means Clustering**: A clustering technique to identify patterns within the data, although primarily used for exploratory analysis.
- **Model Evaluation**: The models are evaluated based on Mean Squared Error (MSE) and R² scores for regression models, and Silhouette Score for clustering.
- **Visualizations**: Correlation matrix, boxplots, feature importance plots, residuals plots, and predicted vs actual price plots are generated to visualize the model’s performance.

## Key Features of the Code

1. **Data Preprocessing**:

   - Missing numerical values are filled with the median.
   - Missing categorical values are filled with the label 'Unknown.'
   - One-hot encoding is applied to categorical variables to convert them into numerical form.

2. **Models Used**:

   - **Random Forest Regressor**: Evaluated using Mean Squared Error and R² score.
   - **Gradient Boosting Regressor**: Evaluated using Mean Squared Error and R² score.
   - **K-Means Clustering**: Evaluated using Silhouette Score.

3. **Data Visualizations**:

   - **Correlation Matrix**: Displays correlations between numerical features.
   - **Boxplot (Region vs Price)**: Visualizes the distribution of property prices across different regions.
   - **Feature Importance Plot**: Displays the most important features affecting property prices in the Random Forest model.
   - **Residual Plots**: Show the difference between actual and predicted values for both Random Forest and Gradient Boosting models.

4. **Performance Metrics**:
   - **Random Forest Regressor**: MSE = 0.0007, R² Score = 0.8196.
   - **Gradient Boosting Regressor**: MSE = 0.0008, R² Score = 0.7707.
   - **K-Means Clustering**: Silhouette Score = 0.2159.

## How to Run

1. Install required Python libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Run the script:

   ```bash
   python assign2.py
   ```

   Ensure that the `Property Sales of Melbourne City.csv` file is in the same directory as the script.

## Results

- **Random Forest Regressor**: Achieved a low mean squared error and high R² score, indicating strong performance in predicting property prices.
- **Gradient Boosting Regressor**: Performed similarly to Random Forest, though slightly less accurate.
- **K-Means Clustering**: Used to explore patterns in the data but was not ideal for predicting continuous variables like property prices.

## Conclusion

This project demonstrates the successful application of machine learning models to predict property prices based on historical sales data. The use of Random Forest and Gradient Boosting Regressors provided accurate price predictions, while K-Means Clustering helped to explore underlying patterns in the dataset.
