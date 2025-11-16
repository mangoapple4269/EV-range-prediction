# EV Range Prediction

This project predicts the estimated driving range of electric vehicles based on key specifications such as top speed, battery capacity, and torque.  
The model is trained using Linear Regression and deployed via a Streamlit app along with an AI chatbot.

## Files

- `main.ipynb` : Jupyter/Colab notebook with data analysis, model training, and evaluation.
- `electric_vehicles_spec_2025.csv` : Dataset used for exploration and training.
- `model.pkl` : Saved Linear Regression model used in the deployed app.
- `app.py` : Streamlit application for prediction and chatbot.
- `requirements.txt` : Required packages for deployment.

## How to Run

1. Open `main.ipynb` in Jupyter or Google Colab.
2. Install required libraries used in the notebook (pandas, seaborn, scikit-learn).
3. Run the notebook step by step to explore data, view visualizations, and train the model.
4. If the model is retrained, save it using:
   ```python
   import joblib
   joblib.dump(model, "model.pkl")


