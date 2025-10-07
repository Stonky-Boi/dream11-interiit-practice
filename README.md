# Dream11 Inter-IIT Tech Meet 13.0 - Team Builder with Predictive AI

## 🎯 Project Overview

AI-powered fantasy cricket team builder that predicts player performance and recommends optimal Dream11 teams using ensemble machine learning models (XGBoost, LightGBM, CatBoost).

### Key Features
✅ Automated data download from Cricsheet (7,692 ODI + T20 matches)  
✅ Comprehensive feature engineering (40+ features)  
✅ Ensemble ML models with weighted predictions  
✅ Interactive Streamlit UI (Product + Model interfaces)  
✅ Audio summaries with text-to-speech  
✅ Real-time visualizations with Plotly  
✅ Strict compliance with 2024-06-30 cutoff  
✅ Dream11 constraint-based team selection  
✅ CSV export for evaluation  

## 🏗️ Project Structure

```
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── main_app.py                      # Main entry point
├── data/
│   ├── raw/
│   │   └── cricksheet_data/        # Raw Cricsheet data (ODIs & T20s)
│   ├── interim/                     # Intermediate processed data
│   └── processed/                   # Final feature-engineered data
├── data_processing/
│   ├── data_download.py             # Download Cricsheet data
│   └── feature_engineering.py       # Feature engineering pipeline
├── model/
│   ├── train_model.py               # Model training (ensemble)
│   └── predict_model.py             # Prediction module
├── model_artifacts/                 # Trained models (XGBoost, LightGBM, CatBoost)
├── UI/
│   ├── product_ui.py                # Product UI (user-facing)
│   └── model_ui.py                  # Model UI (evaluation)
├── out_of_sample_data/              # Test data (post-submission)
└── docs/
    └── video_demo/                  # Demo videos
```

## 🚀 Quick Start

### 1. Download and Process Data

```
# Download ODIs and T20s from Cricsheet (~7,692 matches)
# IMPORTANT: Automatically filters to 2024-06-30 cutoff
python data_processing/data_download.py
```

Expected output:
- `data/interim/matches_raw.csv`: Match-level data
- `data/interim/balls_raw.csv`: Ball-by-ball data
- Validation confirms no data after 2024-06-30

### 2. Engineer Features

```
# Create 40+ features from raw data
python data_processing/feature_engineering.py
```

Creates:
- Historical features (rolling averages, EMA, career stats)
- Venue-specific features
- Opposition-specific features
- Player role classification
- Contextual features (toss, seasonality)

Output: `data/processed/training_data_2024-06-30.csv`

### 3. Train Models

```
# Train ensemble models (XGBoost, LightGBM, CatBoost)
python model/train_model.py
```

Trains 3 models and creates weighted ensemble. Models saved to `model_artifacts/`.

### 4. Launch Application

```
streamlit run main_app.py
```

Opens browser with:
- **Product UI**: Generate Dream11 teams
- **Model UI**: Train custom models and evaluate

## 🎮 Usage

### Product UI - Team Builder

1. Select two teams from dropdown
2. Choose match date
3. Click "Generate Dream Team"
4. View:
   - Top 11 players with predicted points
   - Detailed player explanations
   - Audio team summary
   - Visual analytics

**Features:**
- Sub-10 second response time
- Dream11 constraint compliance
- Role-based team composition
- Interactive visualizations

### Model UI - Evaluation

1. Configure training period (MUST end by 2024-06-30)
2. Set testing period
3. Click "Train & Evaluate"
4. View:
   - Training metrics (MAE, RMSE, R²)
   - Match-by-match predictions
   - MAE distribution
   - Predicted vs actual dream teams
5. Download CSV results

**Output Format:**
```
Match_ID, Match_Date, Team_1, Team_2, Dream_Team_Total_Points,
Predicted_Team_Total, MAE, Predicted_Player_1, ..., Dream_Team_Player_1, ...
```

## 🧠 Model Architecture

### Data Sources
- **ODIs**: 3,019 matches (male + female)
- **T20s**: 4,673 matches (male + female)
- **Total**: ~7,692 matches
- **Cutoff**: 2024-06-30 (strict enforcement)

### Feature Engineering (40+ Features)

**Historical Performance:**
- Rolling averages (3, 5, 10 matches)
- Exponential moving averages
- Career statistics
- Form trends & consistency

**Context-Specific:**
- Venue performance
- Opposition records
- Match format (ODI vs T20)
- Toss impact

**Player Attributes:**
- Role classification (Batsman/Bowler/All-Rounder/WK)
- Batting stats (runs, SR, boundaries)
- Bowling stats (wickets, economy, maidens)
- Fielding stats (catches, stumpings)

### ML Models

**Ensemble Approach:**
1. **XGBoost**: Gradient boosting with 500 trees
2. **LightGBM**: Fast categorical support
3. **CatBoost**: Native categorical handling

**Weighted Ensemble:**
- Inverse MAE weighting
- Best model gets highest weight
- Typical performance: MAE ~15-25 points

### Fantasy Points Calculation

Official Dream11 scoring:
- **Batting**: Runs (1pt), 4s (+1), 6s (+2), 50 (+8), 100 (+16)
- **Bowling**: Wickets (25pt), LBW/Bowled (+8), Maidens (+12)
- **Fielding**: Catches (8pt), Stumpings (12pt), Run-outs (12pt)
- **Bonuses**: Strike rate, economy rate bonuses/penalties

## 📊 Model Performance

Typical metrics on validation set:
- **MAE**: 15-25 fantasy points
- **R²**: 0.65-0.75
- **RMSE**: 20-30 points

Evaluation on test matches provides match-level MAE for dream team total points.

## ⚠️ Critical Compliance Rules

### 1. Data Cutoff (MANDATORY)
```
# ALWAYS enforced in all scripts
train_end_date = '2024-06-30'
```
**Violation = Automatic Disqualification**

### 2. Dream11 Constraints
- 11 players total
- 1-7 players from each team
- 1-4 Wicket-Keepers (min 1)
- 1-8 Batsmen (min 1)
- 1-4 All-Rounders (min 1)
- 1-8 Bowlers (min 1)

### 3. Performance Requirements
- Product UI: < 10 seconds response
- Model UI: Reproducible CSV output
- Explainability: Why each player was selected

## 🛠️ Advanced Usage

### Custom Model Training

```
from model.train_model import Dream11ModelTrainer

trainer = Dream11ModelTrainer(data_path='data/processed/training_data_2024-06-30.csv')
trainer.train_full_pipeline(
    train_end_date='2024-06-30',
    model_name='CustomModel'
)
```

### Making Predictions

```
from model.predict_model import Dream11Predictor

predictor = Dream11Predictor(model_name='ProductUIModel')
predictions = predictor.predict(player_features_df)
```

### Feature Importance

```
importance = predictor.get_feature_importance(model_type='xgboost', top_n=10)
print(importance)
```

## 📝 Evaluation Criteria

Based on Inter-IIT judging:

**ML Model (60%)**
- Accuracy (MAE on dream team predictions)
- Explainability (SHAP, feature importance)
- Creativity & approach

**UI (20%)**
- Intuitiveness & design
- Audio/video integration
- User experience

**Documentation (20%)**
- Code clarity
- README completeness
- Presentation quality

## 🎬 Demo Video

Located in `docs/video_demo/` - showcases:
1. Data pipeline execution
2. Model training process
3. Product UI demo
4. Model UI evaluation
5. Results analysis

## 🔧 Troubleshooting

**Issue: Models not found**
```
python model/train_model.py
```

**Issue: No training data**
```
python data_processing/data_download.py
python data_processing/feature_engineering.py
```

**Issue: Streamlit error**
```
pip install --upgrade streamlit
streamlit run main_app.py
```

**Issue: Date validation fails**
- Check that all data processing respects 2024-06-30 cutoff
- Run validation: `python data_processing/data_download.py` (includes validation)

## 📚 References

- **Data Source**: [Cricsheet](https://cricsheet.org) - Open cricket data
- **Dream11**: Official fantasy scoring rules
- **Problem Statement**: Inter IIT Tech Meet 13.0 Mid-Prep

## 👥 Contributors

IIT Indore - Dream11 Inter-IIT Tech Meet 13.0

## 📄 License

MIT License

---

**Built with ❤️ for Inter IIT Tech Meet 14.0 Practice**