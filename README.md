# **Sports Popularity Analysis and Forecasting**

## **📌 Overview**
This project explores and forecasts the popularity trends of various sports using data from **Google Trends**. It applies advanced **time series analysis** and **machine learning** techniques to model and predict future trends in sports such as **Football, Baseball, Tennis, Table Tennis**, and **Volleyball**.

## **📂 Dataset**
The dataset (`major_events_full_updated.csv`) is sourced from Google Trends and includes:
- Historical monthly popularity data (from 2004 onwards)
- Annotations of major sporting events
- YouTube viewership scores (when available)

## **Features**

### 🕒 Time Series Analysis
- Hybrid ensemble forecasting using **Prophet** and **SARIMA**
- Anomaly detection and outlier handling
- Seasonal-Trend decomposition (STL)
- Stationarity tests (ADF, KPSS)

### 📊 Interactive Visualization
- Dynamic **Plotly** visualizations
- Star markers for major events
- Forecast confidence intervals
- Clear distinction between training and test data

### 🧠 Model Components
- **Prophet Models**: Tailored per sport (`/Model/prophet_model_*.pkl`)
- **SARIMA Models**: Built individually (`/Model/sarima_model_*.pkl`)
- **Ensemble Strategy**: Averages model predictions for improved accuracy

## **🧾 Project Structure**
```
├── DataSets/
│   └── major\_events\_full\_updated.csv
├── Model/
│   ├── prophet\_model\_*.pkl
│   └── sarima\_model\_*.pkl
├── plots/
│   ├── sports\_plot.html
│   ├── Football\_plot.html
│   └── sarima\_forecast.html
├── gui.py
└── main.ipynb
```

## **⚙️ Technical Highlights**
- **Streamlit GUI**: Dark-themed UI with intuitive controls
- **Data Cleaning**: Smart handling of missing values and outliers
- **Evaluation Metrics**: MAE, MSE, R²
- **Interactivity**: Choose sport and forecast duration dynamically

## **🧩 Dependencies**
```bash
streamlit
prophet
pandas
numpy
plotly
statsmodels
scikit-learn
pmdarima
````

## **Getting Started**
```bash
streamlit run gui.py
```

## **📈 Analysis Capabilities**
* Long-term trend insights
* Seasonality detection
* Measuring event impact (e.g. Olympics, World Cups)
* Comparing popularity across sports
* Predictive modeling for future trends

## **📷 Visualization Samples**
* Forecast plots with confidence bands
* STL seasonal decomposition
* Cross-sport trend comparisons
* Impact visualization of major events

## **Future Enhancements**
* Adding more real-world datasets (e.g., Twitter, YouTube trends)


* Automated real-time updates
* Enhanced anomaly detection
* Advanced ensemble methods
* Cross-platform deployment (e.g., Docker, Hugging Face Spaces)

* shared 

