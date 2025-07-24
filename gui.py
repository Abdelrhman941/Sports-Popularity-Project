import os
import io
import warnings
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox, zscore
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore", category=FutureWarning)
# ------------------------------------ Streamlit Interface ------------------------------------
st.set_page_config(
    page_title="Sports Popularity Analysis",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded" )

st.markdown("""
<style>
    .stApp { background-color: #121212; }
    h1, h2, h3 { color: #8ab4f8 !important; font-weight: 700; letter-spacing: 0.5px;}
    .stSidebar .sidebar-content {
        background-color: #1e1e1e;
        border-radius: 8px;
        box-shadow: 0 4px 24px rgba(138,180,248,0.08);
        border: 1.5px solid #222;
        margin: 1rem 0.5rem 1rem 0.5rem;
        padding: 1.5rem 1rem;
    }
    .stRadio > div { flex-direction: row; gap: 1rem; }
    .stRadio label { 
        background-color: #2d2d2d; 
        padding: 1px 18px; 
        border-radius: 10px; 
        margin: 10px 0; 
        transition: all 0.5s ease; 
        color: #e0e0e0; 
        border: 2px solid #222; 
        font-size: 1.05rem;
    }
    .stRadio label:hover, .stRadio label[data-selected="true"] { 
        background-color: #3d3d3d; 
        border-color: #8ab4f8; 
        box-shadow: 0 4px 10px rgba(138,180,248,0.13); 
        color: #8ab4f8;
    }
    .stSelectbox > div > div { 
        background-color: #2d2d2d; 
        color: #e0e0e0; 
        border: 1px solid #222; 
        border-radius: 10px; 
        transition: all 0.3s ease; 
        font-size: 1.05rem;
    }
    .stSelectbox > div > div:hover { 
        background-color: #3d3d3d; 
        border-color: #8ab4f8; 
        box-shadow: 0 4px 10px rgba(138,180,248,0.13); 
    }
    .stButton > button {
        background-color: #2d2d2d;
        color: #8ab4f8;
        border: 2px solid #8ab4f8;
        border-radius: 6px;
        padding: 10px 22px;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #8ab4f8;
        color: #121212;
        border: 2px solid #8ab4f8;
    }
    .stDownloadButton > button {
        background: linear-gradient(270deg, #6a11cb, #2575fc, #6a11cb);
        background-size: 600% 600%;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
        animation: gradientAnimation 8s ease infinite;
        box-shadow: 0 4px 15px rgba(101, 41, 255, 0.4);
    }
    .stDownloadButton > button:hover {
        background-color: #000000;
        animation: none;
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
        color: #8ab4f8;
    }
    @keyframes gradientAnimation {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }
    .stDataFrame { background-color: #2d2d2d; }
    .stDataFrame table { color: #e0e0e0; }
    .stDataFrame th { background-color: #3d3d3d; color: #8ab4f8; }
    .stDataFrame td { background-color: #2d2d2d; }
    .stExpander { 
        background-color: #232323; 
        border-radius: 6px; 
        border: 2px solid #222; 
        transition: all 0.3s ease; 
    }
    .stExpander:hover { border-color: #8ab4f8; box-shadow: 0 4px 10px rgba(138,180,248,0.13); }
    .stExpander summary { color: #8ab4f8; }
    .stMetric { 
        background-color: #232323; 
        border-radius: 7px; 
        padding: 12px; 
        border: 2px solid #222; 
        transition: all 0.3s ease; 
        margin-bottom: 1rem;
    }
    .stMetric:hover { background-color: #3d3d3d; border-color: #8ab4f8; box-shadow: 0 4px 10px rgba(138,180,248,0.13); }
    .stMetric label { color: #8ab4f8 !important; }
    .stMetric p { color: #e0e0e0 !important; }
</style> """, unsafe_allow_html=True)

st.title("Sports Popularity Analysis")
# ------------------------------------ Preprocessing Functions & Helper Functions ------------------------------------
@st.cache_data
def load_data(data_source):
    try:
        df = pd.read_csv('./DataSets/major_events_full_updated.csv', parse_dates=['Date'])
        if data_source not in df.columns:
            st.error(f"Column {data_source} not found in the dataset!")
            st.stop()
        data_pivot = df.pivot(index='Date', columns='Sport', values=data_source)
        data_pivot = data_pivot.sort_index()
        data_pivot = data_pivot.ffill().bfill()
        return data_pivot, df
    except FileNotFoundError:
        st.error("Dataset file not found! Please ensure './DataSets/major_events_full_updated.csv' exists.")
        st.stop()

def preprocess_data(series, test_start_year):
    """Prepare data for forecasting by handling missing values and splitting into train/test."""
    df_prophet = series.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Handle missing values
    if df_prophet['y'].isnull().sum() > 0:
        st.warning("Data contains missing values. Filling with forward-fill and back-fill.")
        df_prophet['y'] = df_prophet['y'].ffill().bfill()
    
    # Split data
    train_df = df_prophet[df_prophet['ds'].dt.year < test_start_year].copy()
    test_df = df_prophet[df_prophet['ds'].dt.year >= test_start_year].copy()
    
    return df_prophet, train_df, test_df

def clean_training_data(train_df):
    """Remove outliers from training data using IsolationForest and z-score fallback."""
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(train_df[['y']])
    train_df_clean = train_df[outliers == 1].copy()
    
    # Fallback to z-score if too many outliers removed
    if len(train_df_clean) < 0.7 * len(train_df):
        st.warning(f"Too many outliers removed ({len(train_df) - len(train_df_clean)}). Using z-score method.")
        z_scores = zscore(train_df['y'])
        train_df_clean = train_df[abs(z_scores) < 3].copy()
    
    return train_df_clean

def create_holidays(original_df, selected_sport):
    """Create holidays dataframe for Prophet based on major events."""
    major_events = original_df[original_df['Sport'] == selected_sport]
    major_events = major_events[major_events['Major_Event'] == 1]['Date'].unique()
    
    holidays = pd.DataFrame({
        'holiday': 'major_event',
        'ds': pd.to_datetime(major_events),
        'lower_window': -2,
        'upper_window': 2 })
    
    return holidays, major_events

def make_stationary(series):
    if len(series.dropna()) < 3:
        return series, 0, series, 'Original'
    
    original_series = series.copy()
    diff_count = 0
    results = {}
    
    results['Original'] = adfuller(series.dropna(), autolag='AIC')[1]
    diff_1 = series.diff().dropna()
    results['Differencing 1'] = adfuller(diff_1, autolag='AIC')[1]
    diff_2 = diff_1.diff().dropna()
    results['Differencing 2'] = adfuller(diff_2, autolag='AIC')[1]
    
    if not (series <= 0).any():
        log_ts = np.log(series)
        results['Log Transform'] = adfuller(log_ts, autolag='AIC')[1]
    else:
        results['Log Transform'] = np.nan
    
    sq_ts = series ** 2
    results['Square Transform'] = adfuller(sq_ts, autolag='AIC')[1]
    
    if not (series <= 0).any():
        boxcox_ts, _ = boxcox(series.dropna())
        boxcox_series = pd.Series(boxcox_ts, index=series.dropna().index)
        results['Box-Cox Transform'] = adfuller(boxcox_series, autolag='AIC')[1]
    else:
        results['Box-Cox Transform'] = np.nan
    
    x_num = np.arange(len(series))
    coeffs = np.polyfit(x_num, series.values, 1)
    trend = coeffs[0] * x_num + coeffs[1]
    detrended = series.values - trend
    detrended_series = pd.Series(detrended, index=series.index)
    results['Detrending Linear'] = adfuller(detrended_series, autolag='AIC')[1]
    
    window = 12
    mov_avg = series.rolling(window=window, center=True).mean()
    detrended_ma = series - mov_avg
    results['Detrending Moving Average'] = adfuller(detrended_ma.dropna(), autolag='AIC')[1]
    
    stationary_methods = {k: v for k, v in results.items() if pd.notna(v) and v <= 0.05}
    if stationary_methods:
        best_method = min(stationary_methods, key=stationary_methods.get)
    else:
        best_method = 'Differencing 1'
    
    if best_method == 'Original':
        transformed_series = series
        diff_count = 0
    elif best_method == 'Differencing 1':
        transformed_series = diff_1
        diff_count = 1
    elif best_method == 'Differencing 2':
        transformed_series = diff_2
        diff_count = 2
    elif best_method == 'Log Transform':
        transformed_series = log_ts
        diff_count = 0
    elif best_method == 'Square Transform':
        transformed_series = sq_ts
        diff_count = 0
    elif best_method == 'Box-Cox Transform':
        transformed_series = boxcox_series
        diff_count = 0
    elif best_method == 'Detrending Linear':
        transformed_series = detrended_series
        diff_count = 0
    elif best_method == 'Detrending Moving Average':
        transformed_series = detrended_ma.dropna()
        diff_count = 0
    
    return transformed_series, diff_count, original_series, best_method, results

def train_test_split_by_year(series, test_start_year):
    train_series = series[series.index.year < test_start_year]
    test_series = series[series.index.year >= test_start_year]
    return train_series, test_series

def plot_time_series(series, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series,
        mode='lines',
        name='Popularity',
        line=dict(color='blue') ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Popularity",
        template="plotly_white",
        showlegend=True )
    return fig

def plot_major_events(series, sport, data_source, original_df):
    sport_data = original_df[original_df['Sport'] == sport].copy()
    if sport_data.empty:
        st.error(f"No data available for {sport} in the original dataset!")
        return None
    
    major_events = sport_data[sport_data['Major_Event'] == 1].copy()
    if major_events.empty:
        st.warning(f"No Major Events found for {sport}!")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sport_data['Date'],
        y=sport_data[data_source],
        mode='lines',
        name=data_source.replace('_', ' '),
        line=dict(color='blue') ))
    if not major_events.empty:
        fig.add_trace(go.Scatter(
            x=major_events['Date'],
            y=major_events[data_source],
            mode='markers',
            name='Major Events',
            marker=dict(size=10, color='red', symbol='circle') ))
    
    fig.add_shape(
        type="rect",
        x0='2020-03-01',
        x1='2020-08-31',
        y0=0,
        y1=max(sport_data[data_source].max(), 100) if not sport_data[data_source].empty else 100,
        yref="y",
        fillcolor="LightSalmon",
        opacity=0.3,
        layer="below",
        line_width=0 )
    
    fig.update_layout(
        title=f"{data_source.replace('_', ' ')} with Major Events and COVID-19 Period for {sport.replace('_', ' ').title()}",
        xaxis_title='Date',
        yaxis_title=data_source.replace('_', ' '),
        template='plotly_white',
        showlegend=True )
    return fig

def detect_outliers(series, sport_name=""):
    X = series.values.reshape(-1, 1)
    index = series.index
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso.fit_predict(X)
    iso_outliers = (iso_pred == -1)
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_pred = lof.fit_predict(X)
    lof_outliers = (lof_pred == -1)
    
    dbscan = DBSCAN(eps=2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    dbscan_outliers = (dbscan_labels == -1)
    
    z_scores = zscore(series.values, nan_policy='omit')
    z_outliers = np.abs(z_scores) > 2.5
    
    outlier_df = pd.DataFrame({
        'IsolationForest': iso_outliers,
        'LOF': lof_outliers,
        'DBSCAN': dbscan_outliers,
        'Z-Score': z_outliers}, index=index)
    
    outlier_df = outlier_df.fillna(False).infer_objects(copy=False).astype(bool)
    
    outlier_pivot = outlier_df[outlier_df.any(axis=1)].copy()
    outlier_pivot = outlier_pivot.reset_index().rename(columns={'index': 'Date'})
    
    overlap_scores = {}
    for method in outlier_df.columns:
        overlap = 0
        for other in outlier_df.columns:
            if other != method:
                overlap += ((outlier_df[method]) & (outlier_df[other])).sum()
        overlap_scores[method] = overlap
    best_method = max(overlap_scores, key=overlap_scores.get)
    
    return outlier_pivot, best_method, outlier_df

def plot_stl_result(stl_result, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stl_result.observed.index, y=stl_result.observed, name='Observed'))
    fig.add_trace(go.Scatter(x=stl_result.trend.index, y=stl_result.trend, name='Trend'))
    fig.add_trace(go.Scatter(x=stl_result.seasonal.index, y=stl_result.seasonal, name='Seasonal'))
    fig.add_trace(go.Scatter(x=stl_result.resid.index, y=stl_result.resid, name='Residual'))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white")
    return fig

def plot_distribution(series, title):
    fig = px.histogram(
        x=series,
        nbins=30,
        title=title,
        labels={'x': 'Popularity', 'y': 'Count'},
        template="plotly_white")
    fig.update_layout(showlegend=False)
    return fig

def plot_train_test_split(train_series, test_series, sport, data_source):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_series.index,
        y=train_series,
        mode='lines',
        name='Training Data',
        line=dict(color='green')))
    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=test_series,
        mode='lines',
        name='Test Data',
        line=dict(color='red')))
    fig.update_layout(
        title=f"Training vs Test Data for {sport.replace('_', ' ').title()}",
        xaxis_title="Date",
        yaxis_title=data_source.replace('_', ' '),
        template="plotly_white",
        showlegend=True)
    return fig

def download_plot(fig, title, sport, series_type=None):
    png_buffer = io.BytesIO()
    fig.write_image(png_buffer, format="png")
    png_buffer.seek(0)
    
    html_buffer = io.StringIO()
    fig.write_html(html_buffer, full_html=False)
    html_bytes = html_buffer.getvalue().encode("utf-8")
    
    png_filename = f"{title.replace(' ', '_')}_{sport}"
    if series_type:
        png_filename += f"_{series_type}"
    png_filename += ".png"
    
    html_filename = f"{title.replace(' ', '_')}_{sport}"
    if series_type:
        html_filename += f"_{series_type}"
    html_filename += ".html"
    
    st.download_button(
        label="📥 Download Plot as PNG !",
        data=png_buffer,
        file_name=png_filename,
        mime="image/png")
    
    st.download_button(
        label="🌐 Download Plot as HTML!",
        data=html_bytes,
        file_name=html_filename,
        mime="text/html")

def train_sarima_model(series, test_start_year):
    """Train SARIMA model and return model with differencing count."""
    try:
        transformed_series, diff_count, _, _, _ = make_stationary(series[series.index.year < test_start_year])
        sarima_model = auto_arima(
            transformed_series,
            seasonal=True,
            m=12,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            start_P=0, start_Q=0,
            max_P=2, max_D=1, max_Q=2,
            d=None,
            D=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic')
        return sarima_model, diff_count
    except Exception as e:
        st.error(f"Error training SARIMA model: {str(e)}")
        return None, 0

def train_prophet_model(train_df_clean, holidays):
    """Train Prophet model with predefined parameters."""
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            interval_width=0.95)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        model.fit(train_df_clean)
        return model
    except Exception as e:
        st.error(f"Error training Prophet model: {str(e)}")
        return None

def save_models(model, sarima_model, diff_count, model_path, sarima_model_path):
    """Save Prophet and SARIMA models to disk."""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"Prophet model saved to {model_path}")
    except Exception as e:
        st.warning(f"Error saving Prophet model: {str(e)}")
    
    if sarima_model is not None:
        try:
            with open(sarima_model_path, 'wb') as f:
                pickle.dump({'model': sarima_model, 'diff_count': diff_count}, f)
            st.success(f"SARIMA model saved to {sarima_model_path}")
        except Exception as e:
            st.warning(f"Error saving SARIMA model: {str(e)}")

def plot_forecast(series, selected_sport, forecast_year, train_df, test_df, forecast, ensemble_forecast, major_events):
    """Create and display the forecast plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df['ds'], y=train_df['y'], mode='lines', name='Training Data', line=dict(color='#4dd0e1', width=2)))
    fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], mode='lines', name='Test Data', line=dict(color='#ff7043', width=2)))
    fig.add_trace(go.Scatter(x=ensemble_forecast['ds'], y=ensemble_forecast['ensemble_yhat'], mode='lines', name='Ensemble Forecast', line=dict(color='#4caf50', width=3)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='rgba(76, 175, 80, 0.3)', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='rgba(76, 175, 80, 0.3)', dash='dash'), fill='tonexty'))
    
    if len(major_events) > 0:
        event_dates = pd.to_datetime(major_events)
        event_values = [series.loc[series.index[np.abs(series.index - date).argmin()]] for date in event_dates]
        fig.add_trace(go.Scatter(x=event_dates, y=event_values, mode='markers', name='Major Events', marker=dict(size=12, color='red', symbol='star')))
    
    fig.update_layout(
        title=f"Advanced Ensemble Forecast for {selected_sport.replace('_', ' ').title()} up to {forecast_year}",
        xaxis_title="Date",
        yaxis_title="Popularity",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig

def generate_forecasts(model, sarima_model, diff_count, series, test_start_year, forecast_year, train_df, test_df):
    """Generate ensemble forecasts using Prophet and SARIMA, including test predictions."""
    try:
        # Generate future dates
        start_date = pd.to_datetime(f'{forecast_year}-01-01')
        future_dates = pd.date_range(start=start_date, periods=12, freq='MS')
        future = pd.DataFrame({
            'ds': pd.to_datetime(pd.concat([pd.Series(series.index), pd.Series(future_dates)]).drop_duplicates().reset_index(drop=True)) })
        
        # Prophet predictions for future
        forecast = model.predict(future)
        forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
        
        # Prophet predictions for test data
        test_forecast = model.predict(test_df[['ds']])
        test_forecast[['yhat', 'yhat_lower', 'yhat_upper']] = test_forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)
        
        # SARIMA predictions
        sarima_forecast = pd.DataFrame(index=future['ds'])
        sarima_pred = sarima_model.predict(n_periods=len(future), return_conf_int=True, alpha=0.05)
        
        if diff_count > 0:
            last_value = series[series.index.year < test_start_year].iloc[-1]
            sarima_values = [last_value]
            for pred in sarima_pred[0]:
                sarima_values.append(sarima_values[-1] + pred)
            sarima_forecast['sarima_yhat'] = np.array(sarima_values[1:])
        else:
            sarima_forecast['sarima_yhat'] = np.array(sarima_pred[0])
        
        sarima_forecast = sarima_forecast.reset_index().rename(columns={'index': 'ds'})
        
        # Ensemble forecast
        ensemble_forecast = forecast.copy()
        if len(sarima_forecast) >= len(ensemble_forecast):
            ensemble_forecast = ensemble_forecast.merge(sarima_forecast[['ds', 'sarima_yhat']], on='ds', how='left')
            prophet_rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
            sarima_rmse = np.sqrt(mean_squared_error(test_df['y'], sarima_model.predict(len(test_df))))
            total_rmse = prophet_rmse + sarima_rmse
            prophet_weight = sarima_rmse / total_rmse
            sarima_weight = prophet_rmse / total_rmse
            ensemble_forecast['ensemble_yhat'] = prophet_weight * ensemble_forecast['yhat'] + sarima_weight * ensemble_forecast['sarima_yhat']
        else:
            st.warning("SARIMA forecast length is shorter than Prophet forecast. Using Prophet predictions only.")
            ensemble_forecast['ensemble_yhat'] = ensemble_forecast['yhat']
        
        ensemble_forecast['ensemble_yhat'] = ensemble_forecast['ensemble_yhat'].clip(lower=0)
        
        return forecast, ensemble_forecast, test_forecast
    except Exception as e:
        st.error(f"Error generating forecasts: {str(e)}")
        return None, None, None

def display_performance_metrics(model, sarima_model, diff_count, test_df, series, test_start_year, test_forecast):
    """Calculate and display model performance metrics using precomputed test_forecast."""
    try:
        # Use precomputed test_forecast
        test_sarima_pred = sarima_model.predict(len(test_df))
        if diff_count > 0:
            last_train_value = series[series.index.year < test_start_year].iloc[-1]
            test_sarima_values = [last_train_value]
            for pred in test_sarima_pred:
                test_sarima_values.append(test_sarima_values[-1] + pred)
            test_sarima_forecast = np.array(test_sarima_values[1:])
        else:
            test_sarima_forecast = np.array(test_sarima_pred)
        
        if len(test_forecast['yhat'].values) != len(test_sarima_forecast):
            st.error("Mismatch in prediction lengths between Prophet and SARIMA!")
            return
        
        prophet_rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
        sarima_rmse = np.sqrt(mean_squared_error(test_df['y'], test_sarima_forecast))
        total_rmse = prophet_rmse + sarima_rmse
        prophet_weight = sarima_rmse / total_rmse
        sarima_weight = prophet_rmse / total_rmse
        test_ensemble_pred = prophet_weight * test_forecast['yhat'].values + sarima_weight * test_sarima_forecast
        
        mae = mean_absolute_error(test_df['y'], test_ensemble_pred)
        rmse = np.sqrt(mean_squared_error(test_df['y'], test_ensemble_pred))
        smape = 200 * np.mean(np.abs(test_ensemble_pred - test_df['y']) / (np.abs(test_df['y']) + np.abs(test_ensemble_pred) + 1e-10))
        r2 = r2_score(test_df['y'], test_ensemble_pred)
        
        mae_threshold = series.mean() * 0.2
        rmse_threshold = series.std() * 0.5
        smape_threshold = 20
        r2_threshold = 0.7
        
        performance_df = pd.DataFrame({
            'Metric Name': ['MAE', 'RMSE', 'SMAPE', 'R²'],
            'Value' : [mae, rmse, smape, r2],
            'Status': [
                "Good" if mae < mae_threshold else "Bad",
                "Good" if rmse < rmse_threshold else "Bad",
                "Good" if smape < smape_threshold else "Bad",
                "Good" if r2 > r2_threshold else "Bad" ] })
        
        styled_df = performance_df.style.applymap(
            lambda x: 'color: green; font-weight: bold' if x == 'Good' else 'color: red; font-weight: bold',
            subset=['Status'] )
        
        st.subheader("📊 Model Performance")
        st.dataframe(styled_df, use_container_width=True)
        
        if any(status == "Bad" for status in performance_df['Status']):
            st.warning("Some metrics indicate poor performance. Consider tuning the model or adding more data.")
        else:
            st.success("Model performance is satisfactory.")
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")
# ------------------------------------ Sidebar ------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/analytics.png", width=80)
    st.title("📂 Data Source")
    data_source = st.selectbox(
        "Select Data Metric:",
        ["Google_Trends_Popularity", "YouTube_Views_Score"],
        index=0,
        help="Choose the metric for popularity analysis." )
    st.markdown("---")
    
    st.title("⚙️ Analysis Configuration")
    st.subheader("🏅 Sport Selection")
    data_pivot, original_df = load_data(data_source)
    sports = data_pivot.columns.tolist()
    
    if not sports:
        st.error("No sports found in the dataset!")
        st.stop()
    
    selected_sport = st.selectbox(
        "Select Sport:",
        options=sports,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Choose a sport to analyze.")
    
    st.subheader("📈 Analysis Type")
    plot_options = {
        "Original Time Series": "time_series",
        "Preprocessing": "preprocessing",
        "Training & Test Data": "train_test",
        "Forecasting": "forecast" }
    selected_plot = st.radio(
        'Select Analysis Type:',
        options=list(plot_options.keys()),
        index=0,
        help="Choose the type of analysis to display." )
    
    if selected_plot == "Preprocessing":
        st.markdown("---")
        st.title("🔍 Preprocessing Options")
        preprocessing_options = [
            "Show Stationary",
            "Distribution Plot",
            "Plotting Major Events",
            "Show Outliers",
            "Show STL Plot" ]
        preprocessing_choice = st.selectbox(
            "Select Preprocessing Option:",
            options=preprocessing_options,
            index=0,
            help="Choose a preprocessing analysis.")
        
        if preprocessing_choice == "Show STL Plot":
            st.subheader("📊 STL Series Type")
            stl_series_type = st.radio(
                "Select Series for STL Decomposition:",
                ["Original", "Transformed"],
                index=0,
                help="Choose whether to decompose the original or transformed series." )
        else:
            stl_series_type = "Original"
    else:
        preprocessing_choice = None
        stl_series_type = "Original"
    
    def Test_Start_Year():
        st.markdown("---")
        test_start_year = st.number_input(
            "Test Start Year",
            min_value=2004,
            max_value=datetime.now().year + 1,
            value=2022,
            help="The year from which test data begins." )
        return test_start_year
    
    test_start_year = 2022
    if selected_plot in ["Training & Test Data", "Forecasting"]:
        test_start_year = Test_Start_Year()
    
    if selected_plot == "Forecasting":
        forecast_year = st.number_input(
            "Forecast Year",
            min_value=datetime.now().year,
            max_value=datetime.now().year + 10,
            value=datetime.now().year + 1,
            help="The year to forecast popularity for." )
        forecast_periods = 12
    else:
        forecast_year = datetime.now().year + 1
        forecast_periods = 12
    st.markdown("---")
    st.info('Developed by [Abdulrahman Ahmed](https://www.linkedin.com/in/abdelrhman-ahmad-13a52b269)')
# ------------------------------------ Main App ------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Here We Go… Again")
with col2:
    series = data_pivot[selected_sport].dropna()
    if series.empty:
        st.error(f"No data available for {selected_sport}!")
        st.stop()
    st.metric(
        label="Number of Records",
        value=len(series),
        delta=f"{series.index.min().year} - {series.index.max().year}" )

st.markdown("---")

with st.expander("ℹ️ Analysis Overview", expanded=False):
    st.markdown(f"""
    This application analyzes the popularity of sports over time using **{data_source.replace('_', ' ')}** data.
    It employs Prophet models to forecast future trends and provides insights into historical patterns.
    
    **How to Use:**
    1. Select a sport from the sidebar.
    2. Choose an analysis type (e.g., Original Time Series, Preprocessing).
    3. Adjust the test start year for training/test splits or forecasting.
    """)

if selected_plot == "Original Time Series":
    st.markdown(f"**Statistical analysis of {selected_sport.replace('_', ' ').title()} popularity trends over time.**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Popularity", f"{series.mean():.1f}", help="Mean value of the series")
    with col2:
        st.metric("Standard Deviation", f"{series.std():.1f}", help="Variability of the series")
    with col3:
        st.metric("Highest Value", f"{series.max():.1f}", help="Maximum value in the series")
    with col4:
        st.metric("Lowest Value", f"{series.min():.1f}", help="Minimum value in the series")
    
    fig = plot_time_series(series, title=f"{selected_sport.replace('_', ' ').title()} Popularity Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Data Table")
    df = pd.DataFrame({'Date': series.index, 'Popularity': series.values})
    st.dataframe(df, use_container_width=True)
    
    download_plot(fig, "Time Series Plot", selected_sport)

elif selected_plot == "Preprocessing":
    transformed_series, diff_count, original_series, best_method, results = make_stationary(series)
    if preprocessing_choice == "Show Stationary":
        st.header("📊 Stationary Series")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=original_series.index,
            y=original_series,
            mode='lines',
            name='Original Series',
            line=dict(color='blue') ))
        fig.add_trace(go.Scatter(
            x=transformed_series.index,
            y=transformed_series,
            mode='lines',
            name=f'Stationary Series ({best_method})',
            line=dict(color='orange') ))
        fig.update_layout(
            title=f"Original vs Stationary Series for {selected_sport.replace('_', ' ').title()}",
            xaxis_title="Date",
            yaxis_title=data_source.replace('_', ' '),
            legend=dict(x=0, y=1),
            template="plotly_white" )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
            <div style='
                background-color: #f0f2f6;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                color: #333;
                border-left: 6px solid #4a90e2;
            '>
                ✅ Best method: <span style='color: #4a90e2;'>{best_method}</span>
                <br>
                📉 p-value: <span style='color: #d14'>{results[best_method]:.4f}</span>
            </div> """, unsafe_allow_html=True)
        st.text("")
        download_plot(fig, "Original_vs_Stationary_Series", selected_sport)
    
    elif preprocessing_choice == "Distribution Plot":
        st.header("📊 Distribution Plot")
        fig = plot_distribution(series, title=f"Distribution of {selected_sport.replace('_', ' ').title()} Popularity")
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "Distribution_Plot", selected_sport)
    
    elif preprocessing_choice == "Plotting Major Events":
        st.header("📅 Plotting Major Events")
        if 'Major_Event' not in original_df.columns:
            st.error("Major_Event column not found in the dataset!")
        else:
            fig = plot_major_events(series, selected_sport, data_source, original_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                download_plot(fig, "Plotting_Major_Events", selected_sport)
    
    elif preprocessing_choice == "Show Outliers":
        st.header("🔍 Show Outliers")
        outlier_pivot, best_method, outlier_df = detect_outliers(series, sport_name=selected_sport.replace('_', ' ').title())
        st.subheader("Outlier Table")
        st.dataframe(outlier_pivot, use_container_width=True)
        st.markdown(f"""
            <div style='
                background-color: #f0f2f6;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                color: #333;
                border-left: 6px solid #4a90e2;
                margin-bottom: 1rem;
            '>
                🏆 Most consistent outlier detection algorithm: 
                <span style='color: #4a90e2;'>{best_method}</span>
            </div> """, unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series,
            mode='lines+markers',
            name='Original Series',
            marker=dict(color='blue', size=6) ))
        colors = {'IsolationForest': 'red', 'LOF': 'green', 'DBSCAN': 'orange', 'Z-Score': 'purple'}
        for method in outlier_df.columns:
            mask = outlier_df[method]
            fig.add_trace(go.Scatter(
                x=series.index[mask],
                y=series[mask],
                mode='markers',
                name=f'Outlier - {method}',
                marker=dict(color=colors[method], size=12, symbol='x') ))
        fig.update_layout(
            title=f"Outlier Detection in {selected_sport.replace('_', ' ').title()}",
            xaxis_title='Date',
            yaxis_title=data_source.replace('_', ' '),
            template='plotly_white' )
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "Outlier_Detection", selected_sport)
    
    elif preprocessing_choice == "Show STL Plot":
        st.header("📈 STL Decomposition Plot")
        series_to_decompose = series if stl_series_type == "Original" else transformed_series
        stl = STL(series_to_decompose, period=12)
        stl_result = stl.fit()
        fig = plot_stl_result(stl_result, title=f"STL Decomposition ({stl_series_type} Series) - {selected_sport.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)
        download_plot(fig, "STL_Decomposition", selected_sport, stl_series_type)

elif selected_plot == "Training & Test Data":
    st.markdown(f"**Splitting {selected_sport.replace('_', ' ').title()} popularity data for training and testing.**")
    
    train_series, test_series = train_test_split_by_year(series, test_start_year)
    
    if train_series.empty or test_series.empty:
        st.error("One of the splits (Training or Test) is empty! Adjust the Test Start Year.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Training Data:** From {train_series.index.min().year} to {test_start_year-1}")
            st.metric("Training Records", len(train_series))
        with col2:
            st.info(f"**Test Data:** From {test_start_year} to {test_series.index.max().year}")
            st.metric("Test Records", len(test_series))
        
        tab1, tab2 = st.tabs(["📈 Combined Chart" ,"📊 Separate Charts"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_series.index, y=train_series, mode='lines', name='Training Data', line=dict(color='#4dd0e1', width=2)))
            fig.add_trace(go.Scatter(x=test_series.index, y=test_series, mode='lines', name='Test Data', line=dict(color='#ff7043', width=2)))
            fig.update_layout(
                title={'text': f"Split of {selected_sport.replace('_', ' ').title()} Data into Training and Test", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=20, color='#8ab4f8')},
                xaxis_title='Date',
                yaxis_title='Popularity Index',
                template='plotly_dark',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='rgba(26,26,26,0.8)',
                paper_bgcolor='rgba(26,26,26,0.8)',
                font=dict(color='#e0e0e0') )
            fig.add_vline(x=test_series.index[0], line_width=1, line_dash="dash", line_color="#aaaaaa")
            st.plotly_chart(fig, use_container_width=True)
            
            download_plot(fig, "Training_vs_Test_Data", selected_sport)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_time_series(train_series, title="Training Data"), use_container_width=True)
            with col2:
                st.plotly_chart(plot_time_series(test_series, title="Test Data"), use_container_width=True)

elif selected_plot == "Forecasting":
    st.markdown(f"**Forecasting {selected_sport.replace('_', ' ').title()} popularity using Ensemble Approach**")
    
    # Define model paths
    os.makedirs('./Model', exist_ok=True)
    model_path = os.path.join('./Model', f'prophet_model_{selected_sport}.pkl')
    sarima_model_path = os.path.join('./Model', f'sarima_model_{selected_sport}.pkl')
    
    # Initialize model variables
    model = None
    sarima_model = None
    diff_count = 0
    need_training = False
    
    # Load existing models
    with st.expander("Model Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    st.success("✅ Prophet model loaded successfully")
                except Exception as e:
                    st.error(f"❌ Error loading Prophet model: {str(e)}")
                    need_training = True
            else:
                st.info("ℹ️ No saved Prophet model found")
                need_training = True
        
        with col2:
            if os.path.exists(sarima_model_path):
                try:
                    with open(sarima_model_path, 'rb') as f:
                        sarima_data = pickle.load(f)
                        sarima_model = sarima_data['model']
                        diff_count = sarima_data['diff_count']
                    st.success("✅ SARIMA model loaded successfully")
                except Exception as e:
                    st.error(f"❌ Error loading SARIMA model: {str(e)}")
                    need_training = True
            else:
                st.info("ℹ️ No saved SARIMA model found")
                need_training = True
    
    # Prepare data
    df_prophet, train_df, test_df = preprocess_data(series, test_start_year)
    
    if train_df.empty or test_df.empty:
        st.error("⚠️ One of the splits (Training or Test) is empty! Adjust the Test Start Year.")
    else:
        # Create holidays
        holidays, major_events = create_holidays(original_df, selected_sport)
        
        # Train models if needed
        if need_training:
            with st.status("Training models...", expanded=True) as status:
                status.update(label="Preprocessing data...")
                train_df_clean = clean_training_data(train_df)
                
                status.update(label="Training SARIMA model...")
                sarima_model, diff_count = train_sarima_model(series, test_start_year)
                
                if sarima_model is not None:
                    status.update(label="Training Prophet model...")
                    model = train_prophet_model(train_df_clean, holidays)
                
                if model is not None and sarima_model is not None:
                    save_models(model, sarima_model, diff_count, model_path, sarima_model_path)
                    status.update(label="Training complete", state="complete")
                else:
                    status.update(label="Training failed", state="error")
                    st.error("Failed to train models. Please check the errors above.")
        
        # Generate and display forecasts
        if model is not None and sarima_model is not None:
            forecast, ensemble_forecast, test_forecast = generate_forecasts(model, sarima_model, diff_count, series, test_start_year, forecast_year, train_df, test_df)
            
            if forecast is not None and ensemble_forecast is not None and test_forecast is not None:
                # Plot results
                fig = plot_forecast(series, selected_sport, forecast_year, train_df, test_df, forecast, ensemble_forecast, major_events)
                
                # Display forecast data
                forecast_year_df = ensemble_forecast[ensemble_forecast['ds'].dt.year == forecast_year][['ds', 'ensemble_yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Date',
                    'ensemble_yhat': 'Forecast',
                    'yhat_lower': 'Lower CI',
                    'yhat_upper': 'Upper CI' })
                st.subheader(f"📋 Forecast Data for {forecast_year}")
                st.dataframe(forecast_year_df, use_container_width=True)
                
                # Display performance metrics
                display_performance_metrics(model, sarima_model, diff_count, test_df, series, test_start_year, test_forecast)
                
                # Download plot
                download_plot(fig, "Time Series Forecasting", selected_sport)
                
                # Download models
                try:
                    with io.BytesIO() as buffer:
                        pickle.dump(model, buffer)
                        buffer.seek(0)
                    
                    with io.BytesIO() as buffer:
                        pickle.dump({'model': sarima_model, 'diff_count': diff_count}, buffer)
                        buffer.seek(0)
                    
                    # Save to session state
                    st.session_state['model'] = model
                    st.session_state['test_df'] = test_df
                    st.session_state['test_forecast'] = test_forecast
                except Exception as e:
                    st.error(f"Error providing model downloads: {str(e)}")
        else:
            st.error("Both Prophet and SARIMA models are required for forecasting. Please ensure models are trained or loaded.")

else:
    st.write("Select an analysis type from the sidebar.")

st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    © 2023 Sports Popularity Analysis | All Rights Reserved
</div>""", unsafe_allow_html=True)