import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
import itertools

# Optional PyTorch imports for ProtoPNet
try:
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. ProtoPNet mode disabled.")

warnings.filterwarnings('ignore')


# ========== ProtoPNet Utilities (from Protopnetcode.py) ==========

PROTOTYPE_LABELS = {
    0: ("Good Wicket Keeper-Batsman (Balanced)", "High weighting on batting and fielding"),
    1: ("Average Batting Allrounder (Lower Order)", "Mix of batting fantasy points and bowling"),
    2: ("Good Bowling Allrounder (Economical)", "Strong bowling control with minor batting"),
    3: ("Keeper-Allrounder (Defensive Field Focus)", "High stumpings/runouts and decent batting"),
    4: ("Technical Batsman (Consistent Scorer)", "Dominant batting features"),
    5: ("Elite Keeper-Batsman (Run-Heavy)", "Heavy batting plus consistent stumpings/catches"),
    6: ("Explosive Batting Allrounder", "High batting + bowling wickets"),
    7: ("Fielding-Oriented Batsman (Support Role)", "Emphasis on fielding and moderate batting"),
    8: ("Aggressive Batting Allrounder", "Batting strike rate and boundary frequency"),
    9: ("Power Hitter (Boundary Specialist)", "Focused on sixes/fours and strike rate"),
    10: ("Strike Bowler (Attack-Driven)", "Strong bowling wickets with minor batting"),
    11: ("Control Bowler (Economy-Driven)", "Bowling economy and dots dominate"),
    12: ("Aggressive Bowling Allrounder", "High bowling wickets with batting support"),
    13: ("Spin/Utility Bowler (Control and Support)", "Bowling economy as core feature"),
    14: ("Fielding Allrounder (Balanced Role)", "High fielding involvement with moderate batting")
}

def get_domain_indices():
    """Get indices for batting, bowling, fielding features across all time periods"""
    batting, bowling, fielding = [], [], []
    for i in range(4):
        offset = i * 11
        batting += [offset + j for j in [3, 5, 6, 7, 10]]
        bowling += [offset + j for j in [1, 2, 9, 10]]
        fielding += [offset + j for j in [0, 4, 8, 10]]
    return batting, bowling, fielding

def generate_feature_combos(vals):
    """Generate all 11,231 feature combinations for ProtoPNet"""
    batting_idx, bowling_idx, fielding_idx = get_domain_indices()
    combos = []
    
    # Batting: 1-4 combos from 20 features
    for r in range(1, 5):
        for idxs in itertools.combinations(batting_idx, r):
            combos.append(np.sum(vals[list(idxs)]))
    
    # Bowling: 1-4 combos from 16 features
    for r in range(1, 5):
        for idxs in itertools.combinations(bowling_idx, r):
            combos.append(np.sum(vals[list(idxs)]))
    
    # Fielding: 1-4 combos from 16 features
    for r in range(1, 5):
        for idxs in itertools.combinations(fielding_idx, r):
            combos.append(np.sum(vals[list(idxs)]))
    
    # All-stats: sum of all 11 features per period (4 periods)
    for i in range(4):
        offset = i * 11
        all_stats_idx = list(range(offset, offset + 11))
        combos.append(np.sum(vals[all_stats_idx]))
    
    return np.array(combos, dtype=np.float32)

def calculate_mape(actual, predicted):
    """Calculate MAPE excluding zero actual values"""
    mask = actual != 0
    n_samples = mask.sum()
    if n_samples > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape, n_samples
    else:
        return None, 0

if TORCH_AVAILABLE:
    class ProtoPNetModel(nn.Module):
        def __init__(self, input_dim=11231, num_prototypes=15, output_dim=11):
            super().__init__()
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
            self.prediction_head = nn.Sequential(
                nn.Linear(num_prototypes, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
        
        def forward(self, x):
            dists = torch.cdist(x, self.prototypes)
            sims = torch.softmax(-dists, dim=1)
            predictions = self.prediction_head(sims)
            return predictions, sims

# ========== Main Predictor Class ==========

class Dream11Predictor:
    """Unified predictor supporting both traditional ML and ProtoPNet"""
    
    def __init__(self, model_dir='model_artifacts', model_name='ProductUIModel', mode='traditional'):
        """
        Args:
            model_dir: Directory containing trained models
            model_name: Model artifact name prefix
            mode: 'traditional' (XGB/LGBM/CB) or 'protopnet' (PyTorch)
        """
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.data_dir = Path('data')
        self.mode = mode
        
        if mode == 'protopnet' and not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available, falling back to traditional mode")
            self.mode = 'traditional'
        
        self.models = {}
        self.ensemble_weights = {}
        self.metadata = {}
        self.aggregate_data = {}
        
        # ProtoPNet-specific attributes
        self.protopnet_model = None
        self.protopnet_checkpoint = None
        
        # Load aggregate stats for feature calculation
        self.load_aggregate_stats()
        
        # Load models based on mode
        if self.mode == 'traditional':
            self.load_traditional_models()
        elif self.mode == 'protopnet':
            self.load_protopnet_model()
    
    def load_aggregate_stats(self):
        """Load aggregate career statistics for feature calculation"""
        print("Loading aggregate statistics...")
        
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
    
    def load_traditional_models(self):
        """Load traditional ML models (XGBoost, LightGBM, CatBoost)"""
        print("=" * 70)
        print("LOADING TRADITIONAL MODELS")
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
    
    def load_protopnet_model(self, model_path=None, device='cuda'):
        """Load ProtoPNet model"""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available, cannot load ProtoPNet")
            return
        
        print("=" * 70)
        print("LOADING PROTOPNET MODEL")
        print("=" * 70)
        
        if model_path is None:
            model_path = self.model_dir / f"{self.model_name}_protopnet.pth"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"ProtoPNet model not found: {model_path}")
        
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️ CUDA not available, using CPU")
        
        print(f"Loading from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        self.protopnet_checkpoint = checkpoint
        self.feature_cols = checkpoint['feature_cols']
        
        # Initialize and load model
        self.protopnet_model = ProtoPNetModel(
            input_dim=11231, 
            num_prototypes=15, 
            output_dim=11
        ).to(device)
        self.protopnet_model.load_state_dict(checkpoint['model_state_dict'])
        self.protopnet_model.eval()
        
        print(f"✓ Loaded ProtoPNet model with {checkpoint['num_prototypes']} prototypes")
        print(f"✓ Device: {device}")
        print("=" * 70)
    
    def prepare_prediction_features(self, player_features_df):
        """Prepare features for traditional ML models"""
        # Ensure all required features are present
        missing_cols = [col for col in self.feature_cols if col not in player_features_df.columns]
        if missing_cols:
            print(f"⚠️ Missing {len(missing_cols)} features, filling with defaults")
            for col in missing_cols:
                if col in self.categorical_features:
                    player_features_df[col] = 'Unknown'
                elif col.startswith('career_'):
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
    
    def predict_traditional(self, player_features_df):
        """Predict using traditional ML ensemble"""
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
    
    def predict_protopnet(self, player_features_df, device='cuda'):
        """Predict using ProtoPNet model"""
        if not TORCH_AVAILABLE or self.protopnet_model is None:
            print("⚠️ ProtoPNet not available")
            return player_features_df
        
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        # Prepare features
        X = player_features_df[self.feature_cols].values.astype(np.float32)
        
        # Normalize
        X_min = self.protopnet_checkpoint['X_min']
        X_max = self.protopnet_checkpoint['X_max']
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        
        # Generate feature combinations
        print("Generating feature combinations...")
        test_combos = np.array([generate_feature_combos(x) for x in tqdm(X_norm, desc="Computing combos")])
        test_combos_tensor = torch.from_numpy(test_combos).float().to(device)
        
        # Predict
        with torch.no_grad():
            predictions_norm, sims = self.protopnet_model(test_combos_tensor)
            
            # Denormalize
            y_min = self.protopnet_checkpoint['y_min']
            y_max = self.protopnet_checkpoint['y_max']
            predictions = predictions_norm.cpu().numpy() * (y_max - y_min) + y_min
            
            best_prototypes = torch.argmax(sims, dim=1).cpu().numpy()
        
        # Build results
        result_df = player_features_df.copy()
        target_cols = self.protopnet_checkpoint['target_cols']
        
        for i, col in enumerate(target_cols):
            result_df[f'pred_{col}'] = predictions[:, i]
        
        result_df['predicted_fantasy_points'] = predictions[:, 0]  # First target is usually fantasy points
        result_df['best_prototype'] = best_prototypes
        result_df['prototype_label'] = [PROTOTYPE_LABELS[p][0] for p in best_prototypes]
        result_df['prototype_rationale'] = [PROTOTYPE_LABELS[p][1] for p in best_prototypes]
        
        return result_df
    
    def predict(self, player_features_df, device='cuda'):
        """Unified prediction method"""
        if self.mode == 'traditional':
            return self.predict_traditional(player_features_df)
        elif self.mode == 'protopnet':
            return self.predict_protopnet(player_features_df, device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_player_recent_features(self, player_name, match_type='t20', num_recent=10):
        """Get recent match history for a player"""
        training_data_path = self.data_dir / 'processed' / 'training_data_all.csv'
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
        """Calculate features for upcoming match prediction"""
        # Get recent match history
        recent_matches = self.get_player_recent_features(player_name, match_type)
        
        if recent_matches is None or len(recent_matches) == 0:
            print(f"⚠️ No historical data found for {player_name}")
            return None
        
        latest_match = recent_matches.iloc[-1]
        
        # Get aggregate stats
        agg_stats = {}
        if player_name in self.aggregate_data:
            format_key = 'T20' if match_type.lower() == 't20' else 'ODI'
            if format_key in self.aggregate_data[player_name]:
                agg_stats = self.aggregate_data[player_name][format_key]
        
        # Calculate rolling features from recent matches
        rolling_features = {
            'avg_fantasy_points_last_3': recent_matches['fantasy_points'].tail(3).mean(),
            'avg_fantasy_points_last_5': recent_matches['fantasy_points'].tail(5).mean(),
            'avg_fantasy_points_last_10': recent_matches['fantasy_points'].tail(10).mean(),
            'avg_runs_last_3': recent_matches['avg_runs_last_3'].tail(3).mean(),
            'avg_runs_last_5': recent_matches['avg_runs_last_5'].tail(5).mean(),
            'avg_runs_last_10': recent_matches['avg_runs_last_10'].tail(10).mean(),
            'avg_wickets_last_3': recent_matches['avg_wickets_last_3'].tail(3).mean(),
            'avg_wickets_last_5': recent_matches['avg_wickets_last_5'].tail(5).mean(),
            'avg_wickets_last_10': recent_matches['avg_wickets_last_10'].tail(10).mean(),
            'form_trend': latest_match.get('form_trend', 0),
            'ema_fantasy_points': latest_match.get('ema_fantasy_points', 0),
        }
        
        # Create feature dictionary
        features = {
            'player': player_name,
            'match_type': match_type.lower(),
            'venue': venue,
            'opposition': opposition,
            'role': latest_match.get('role', 'All-Rounder'),
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
        }
        
        return pd.DataFrame([features])
    
    def predict_for_squad(self, squad_players, match_type='t20', venue='', team1='', team2=''):
        """
        Predict for entire squad
        
        Args:
            squad_players: List of player names
            match_type: 't20' or 'odi'
            venue: Venue name (optional)
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
    """Test predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dream11 Prediction')
    parser.add_argument('--mode', choices=['traditional', 'protopnet'], default='traditional',
                       help='Prediction mode')
    parser.add_argument('--model-dir', default='model_artifacts', help='Model directory')
    parser.add_argument('--model-name', default='ProductUIModel', help='Model name')
    
    args = parser.parse_args()
    
    predictor = Dream11Predictor(
        model_dir=args.model_dir,
        model_name=args.model_name,
        mode=args.mode
    )
    
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
        display_cols = ['player', 'predicted_fantasy_points', 'role']
        if 'prototype_label' in results.columns:
            display_cols.append('prototype_label')
        print(results[display_cols])

if __name__ == '__main__':
    main()