import streamlit as st
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from datetime import date, timedelta
from UI.utils import solve_team_optimization

def run_evaluation(train_start, train_end, test_start, test_end, data_df, roles_df, model_choice):
    st.write(f"Training **{model_choice}** model on data from **{train_start}** to **{train_end}**...")
    train_df = data_df[(data_df['date'] >= str(train_start)) & (data_df['date'] <= str(train_end))].copy()
    test_df = data_df[(data_df['date'] >= str(test_start)) & (data_df['date'] <= str(test_end))].copy()
    target = 'fantasy_points'
    features = [col for col in train_df.columns if col.startswith('roll_')]
    X_train, y_train = train_df[features], train_df[target]
    if model_choice == 'lightgbm':
        model = lgb.LGBMRegressor(objective='regression', random_state=42)
        model.fit(X_train, y_train)
    elif model_choice == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, 
                                 early_stopping_rounds=50, eval_metric="rmse", random_state=42)
        val_size = int(len(X_train) * 0.1)
        X_train_split, y_train_split = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
        X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
        model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=False)
    st.write("✅ Model training complete.")
    st.write(f"Evaluating model...")
    test_df['predicted_points'] = model.predict(test_df[features])
    results = []
    test_matches = test_df['match_id'].unique()
    progress_bar = st.progress(0, text="Evaluating matches...")
    for i, match_id in enumerate(test_matches):
        match_df = test_df[test_df['match_id'] == match_id].copy()
        match_df['team'] = match_df['player'].apply(lambda x: data_df[data_df['player'] == x]['team'].iloc[0])
        match_df = match_df.merge(roles_df, on='player', how='left').fillna('BAT')
        predicted_team = solve_team_optimization(match_df, points_col='predicted_points')
        actual_team = solve_team_optimization(match_df, points_col='fantasy_points')
        predicted_total = predicted_team['predicted_points'].sum()
        actual_total = actual_team['fantasy_points'].sum()
        mae = abs(actual_total - predicted_total)
        results.append({
            'Match Date': match_df['date'].iloc[0].strftime('%Y-%m-%d'),
            'Total Points Predicted': predicted_total,
            'Total Points Actual Dream Team': actual_total,
            'MAE': mae
        })
        progress_bar.progress((i + 1) / len(test_matches), text=f"Evaluating match {i+1}/{len(test_matches)}")
    st.write("✅ Evaluation complete.")
    return train_df, test_df, pd.DataFrame(results)

def show_page(data_df, roles_df, model_choice):
    """Renders the Model UI page."""
    st.title("📊 Model Performance Evaluator")
    st.write(f"You are currently evaluating the **{model_choice}** model.")
    
    today = date.today()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Period")
        train_start = st.date_input("Start Date", today - timedelta(days=365*2), key='train_start')
        train_end = st.date_input("End Date", date(2024, 6, 30), key='train_end')
    with col2:
        st.subheader("Testing Period")
        test_start = st.date_input("Start Date", date(2024, 7, 1), key='test_start')
        test_end = st.date_input("End Date", today, key='test_end')

    if st.button("🚀 Run Evaluation"):
        if train_end >= test_start:
            st.error("Error: The training period must end before the testing period begins.")
        else:
            train_results, test_results, eval_results = run_evaluation(train_start, train_end, test_start, test_end, data_df, roles_df, model_choice)
            st.session_state.eval_train_results = train_results
            st.session_state.eval_test_results = test_results
            st.session_state.eval_results = eval_results

    if 'eval_results' in st.session_state:
        st.subheader("Evaluation Results")
        st.dataframe(st.session_state.eval_results)
        
        st.subheader("⬇️ Download Evaluation Artifacts")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.download_button(
                label="Training Data",
                data=st.session_state.eval_train_results.to_csv(index=False).encode('utf-8'),
                file_name=f'training_data_{today}.csv',
                mime='text/csv'
            )
        with col_b:
            st.download_button(
                label="Testing Data",
                data=st.session_state.eval_test_results.to_csv(index=False).encode('utf-8'),
                file_name=f'testing_data_{today}.csv',
                mime='text/csv'
            )
        with col_c:
            st.download_button(
                label="Evaluation Results",
                data=st.session_state.eval_results.to_csv(index=False).encode('utf-8'),
                file_name=f'evaluation_results_{model_choice}_{today}.csv',
                mime='text/csv'
            )