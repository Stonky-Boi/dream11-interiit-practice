"""
Prediction Module for Dream11 Fantasy Points
Handles expanded feature set (60+ features) with aggregate stats
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class Dream11Predictor:
    """Make predictions using trained ensemble models"""
    
    def __init__(self, model_dir='model_artifacts', model_name='ProductUIModel'):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.data_dir = Path('data')
        
        self.models = {}
        self.ensemble_weights = {}
        self.metadata = {}
        
        # Load aggregate stats for feature calculation
        self.load_aggregate_stats()
        self.load_models()
    
    def load_aggregate_stats(self):
        """Load aggregate career statistics for feature calculation"""
        print("Loading aggregate statistics...")
        
        self.aggregate_data = {}
        
        # Load ODI aggregate
        odi_agg_path = self.data_dir / 'processed' / 'ODI_ODM_data_aggregate_data.json'
        if odi_agg_path.exists():
            with open(odi_agg_path, 'r') as f:
                odi_agg = json.load(f)
                for player, stats in odi_agg.items():
                    if player not in self.aggregate_data:
                        self.aggregate_data[player] = {'ODI': stats}
                    else:
                        self.aggregate_data[player]['ODI'] = stats
        
        # Load T20 aggregate
        t20_agg_path = self.data_dir / 'processed' / 'T20_data_aggregate_data.json'
        if t20_agg_path.exists():
            with open(t20_agg_path, 'r') as f:
                t20_agg = json.load(f)
                for player, stats in t20_agg.items():
                    if player not in self.aggregate_data:
                        self.aggregate_data[player] = {'T20': stats}
                    else:
                        self.aggregate_data[player]['T20'] = stats
        
        print(f"✓ Loaded aggregate stats for {len(self.aggregate_data):,} players")
    
    def load_models(self):
        """Load all trained models and metadata"""
        print("=" * 70)
        print("LOADING MODELS")
        print("=" * 70)
        
        # Load metadata
        metadata_path = self.model_dir / f"{self.model_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_cols = self.metadata['feature_columns']
        self.categorical_features = self.metadata.get('categorical_features', ['role', 'match_type'])
        
        print(f"✓ Model: {self.model_name}")
        print(f"✓ Features: {len(self.feature_cols)}")
        print(f"✓ Ensemble MAE: {self.metadata.get('ensemble_mae', 0):.2f}")
        
        # Load ensemble weights
        weights_path = self.model_dir / f"{self.model_name}_ensemble_weights.json"
        with open(weights_path, 'r') as f:
            self.ensemble_weights = json.load(f)
        
        # Load individual models
        for model_type in ['xgboost', 'lightgbm', 'catboost']:
            model_path = self.model_dir / f"{self.model_name}_{model_type}.pkl"
            if model_path.exists():
                self.models[model_type] = joblib.load(model_path)
                print(f"✓ Loaded {model_type} (weight: {self.ensemble_weights.get(model_type, 0):.4f})")
        
        print(f"✓ Loaded {len(self.models)} models")
        print("=" * 70)
    
    def prepare_prediction_features(self, player_features_df):
        """
        Prepare features for prediction
        Now includes aggregate career stats
        """
        # Ensure all required features are present
        missing_cols = [col for col in self.feature_cols if col not in player_features_df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing {len(missing_cols)} features, filling with defaults")
            for col in missing_cols:
                if col in self.categorical_features:
                    player_features_df[col] = 'Unknown'
                elif col.startswith('career_'):
                    # Fill career stats with 0
                    player_features_df[col] = 0
                else:
                    player_features_df[col] = 0
        
        # Select only required features in correct order
        X = player_features_df[self.feature_cols].copy()
        
        # Handle categorical features
        for cat_col in self.categorical_features:
            if cat_col in X.columns:
                if X[cat_col].dtype.name != 'category':
                    X[cat_col] = X[cat_col].fillna('Unknown').astype(str)
                    X[cat_col] = pd.Categorical(X[cat_col])
                else:
                    if 'Unknown' not in X[cat_col].cat.categories:
                        X[cat_col] = X[cat_col].cat.add_categories(['Unknown'])
                    X[cat_col] = X[cat_col].fillna('Unknown')
        
        # Handle numerical features
        for col in self.feature_cols:
            if col not in self.categorical_features:
                median_val = X[col].median() if not X[col].isna().all() else 0
                X[col] = X[col].fillna(median_val)
        
        return X
    
    def predict(self, player_features_df):
        """Make ensemble predictions"""
        X = self.prepare_prediction_features(player_features_df)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            ensemble_pred += weight * pred
        
        # Store results
        result_df = player_features_df.copy()
        result_df['predicted_fantasy_points'] = ensemble_pred
        
        for name, pred in predictions.items():
            result_df[f'pred_{name}'] = pred
        
        return result_df
    
    def get_player_recent_features(self, player_name, match_type='t20', num_recent=10):
        """
        Get recent match features for a player from training data
        Used to calculate rolling averages
        """
        training_data_path = self.data_dir / 'processed' / 'training_data_2024-06-30.csv'
        
        if not training_data_path.exists():
            return None
        
        df = pd.read_csv(training_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for player and match type
        player_data = df[
            (df['player'] == player_name) & 
            (df['match_type'] == match_type.lower())
        ].sort_values('date').tail(num_recent)
        
        if len(player_data) == 0:
            return None
        
        return player_data
    
    def calculate_features_for_upcoming_match(self, player_name, match_type='t20', venue='', opposition=''):
        """
        Calculate all 60+ features for a player for an upcoming match
        Uses historical data + aggregate stats
        """
        # Get recent match history
        recent_matches = self.get_player_recent_features(player_name, match_type)
        
        if recent_matches is None or len(recent_matches) == 0:
            print(f"⚠️  No historical data found for {player_name}")
            return None
        
        latest_match = recent_matches.iloc[-1]
        
        # Get aggregate stats
        agg_stats = {}
        if player_name in self.aggregate_data:
            format_key = 'T20' if match_type.lower() == 't20' else 'ODI'
            if format_key in self.aggregate_data[player_name]:
                agg_stats = self.aggregate_data[player_name][format_key]
        
        # Calculate rolling features
        rolling_features = {
            'avg_fantasy_points_last_3': recent_matches['fantasy_points'].tail(3).mean(),
            'avg_fantasy_points_last_5': recent_matches['fantasy_points'].tail(5).mean(),
            'avg_fantasy_points_last_10': recent_matches['fantasy_points'].tail(10).mean(),
            'avg_runs_last_3': recent_matches['total_runs'].tail(3).mean(),
            'avg_runs_last_5': recent_matches['total_runs'].tail(5).mean(),
            'avg_runs_last_10': recent_matches['total_runs'].tail(10).mean(),
            'avg_wickets_last_3': recent_matches['total_wickets'].tail(3).mean(),
            'avg_wickets_last_5': recent_matches['total_wickets'].tail(5).mean(),
            'avg_wickets_last_10': recent_matches['total_wickets'].tail(10).mean(),
            'ema_fantasy_points': recent_matches['fantasy_points'].ewm(span=5, adjust=False).mean().iloc[-1],
            'form_trend': (recent_matches['fantasy_points'].tail(3).mean() - 
                          recent_matches['fantasy_points'].tail(10).mean()),
            'consistency_last_5': recent_matches['fantasy_points'].tail(5).std(),
        }
        
        # Create feature dictionary
        features = {
            'player': player_name,
            'match_type': match_type.lower(),
            'venue': venue,
            'opposition': opposition,
            'role': latest_match.get('role', 'All-Rounder'),
            
            # Use averages from recent matches as proxy
            **rolling_features,
            
            # Career aggregate stats
            'career_matches': float(agg_stats.get('Matches', 0)) if agg_stats else 0,
            'career_innings_batted': float(agg_stats.get('InningsBatted', 0)) if agg_stats else 0,
            'career_innings_bowled': float(agg_stats.get('InningsBowled', 0)) if agg_stats else 0,
            'career_total_runs': float(agg_stats.get('Runs', 0)) if agg_stats else 0,
            'career_batting_avg': float(agg_stats.get('Average', 0)) if agg_stats else 0,
            'career_strike_rate': float(agg_stats.get('StrikeRate', 0)) if agg_stats else 0,
            'career_highest_score': float(agg_stats.get('HighestScore', 0)) if agg_stats else 0,
            'career_fifties': float(agg_stats.get('50s', 0)) if agg_stats else 0,
            'career_hundreds': float(agg_stats.get('100s', 0)) if agg_stats else 0,
            'career_fours': float(agg_stats.get('4s', 0)) if agg_stats else 0,
            'career_sixes': float(agg_stats.get('6s', 0)) if agg_stats else 0,
            'career_total_wickets': float(agg_stats.get('Wickets', 0)) if agg_stats else 0,
            'career_bowling_avg': float(agg_stats.get('BowlingAverage', 0)) if agg_stats else 0,
            'career_economy': float(agg_stats.get('EconomyRate', 0)) if agg_stats else 0,
            'career_bowling_sr': float(agg_stats.get('BowlingStrikeRate', 0)) if agg_stats else 0,
            'career_four_wickets': float(agg_stats.get('4w', 0)) if agg_stats else 0,
            'career_five_wickets': float(agg_stats.get('5w', 0)) if agg_stats else 0,
            'career_catches': float(agg_stats.get('Catches', 0)) if agg_stats else 0,
            'career_stumpings': float(agg_stats.get('Stumpings', 0)) if agg_stats else 0,
            'career_run_outs': float(agg_stats.get('RunOuts', 0)) if agg_stats else 0,
            
            # Other features (use latest values)
            'total_runs': latest_match.get('total_runs', 0),
            'balls_faced': latest_match.get('balls_faced', 0),
            'fours': latest_match.get('fours', 0),
            'sixes': latest_match.get('sixes', 0),
            'strike_rate': latest_match.get('strike_rate', 0),
            'is_duck': 0,
            'total_wickets': latest_match.get('total_wickets', 0),
            'balls_bowled': latest_match.get('balls_bowled', 0),
            'runs_conceded': latest_match.get('runs_conceded', 0),
            'economy_rate': latest_match.get('economy_rate', 0),
            'maidens': latest_match.get('maidens', 0),
            'overs_bowled': latest_match.get('overs_bowled', 0),
            'catches': latest_match.get('catches', 0),
            'stumpings': latest_match.get('stumpings', 0),
            'run_outs': latest_match.get('run_outs', 0),
            'num_innings_batted': latest_match.get('num_innings_batted', 1),
            'avg_runs_per_inning': latest_match.get('avg_runs_per_inning', 0),
            'avg_wickets_per_inning': latest_match.get('avg_wickets_per_inning', 0),
            'avg_sixes_per_inning': latest_match.get('avg_sixes_per_inning', 0),
            'avg_fours_per_inning': latest_match.get('avg_fours_per_inning', 0),
            'avg_balls_faced_per_inning': latest_match.get('avg_balls_faced_per_inning', 0),
            'avg_balls_bowled_per_inning': latest_match.get('avg_balls_bowled_per_inning', 0),
            'boundary_percentage': latest_match.get('boundary_percentage', 0),
            'runs_per_ball': latest_match.get('runs_per_ball', 0),
            'dot_ball_percentage': 0,
            'bowling_strike_rate': latest_match.get('bowling_strike_rate', 0),
            'runs_per_ball_conceded': latest_match.get('runs_per_ball_conceded', 0),
            'dot_balls_bowled': 0,
            'wickets_per_innings': latest_match.get('wickets_per_innings', 0),
            'matches_in_last_30_days': 5,  # Approximate
        }
        
        return pd.DataFrame([features])
    
    def predict_for_squad(self, squad_players, match_type='t20', venue='', team1='', team2=''):
        """
        Predict fantasy points for a squad of players
        
        Args:
            squad_players: List of player names
            match_type: 't20' or 'odi'
            venue: Venue name
            team1: First team name
            team2: Second team name
        """
        predictions = []
        
        for player in squad_players:
            # Determine opposition
            opposition = team2 if player in team1 else team1
            
            # Calculate features
            player_features = self.calculate_features_for_upcoming_match(
                player, match_type, venue, opposition
            )
            
            if player_features is not None:
                # Predict
                result = self.predict(player_features)
                predictions.append(result)
        
        if predictions:
            return pd.concat(predictions, ignore_index=True)
        else:
            return pd.DataFrame()

def main():
    """Test prediction module"""
    predictor = Dream11Predictor()
    
    # Test with sample players
    sample_players = ['V Kohli', 'RG Sharma', 'JJ Bumrah']
    
    print("\nTesting predictions...")
    results = predictor.predict_for_squad(
        sample_players, 
        match_type='t20',
        venue='Wankhede Stadium',
        team1='India',
        team2='Australia'
    )
    
    if len(results) > 0:
        print("\nPredictions:")
        print(results[['player', 'predicted_fantasy_points', 'role']])

if __name__ == '__main__':
    main()