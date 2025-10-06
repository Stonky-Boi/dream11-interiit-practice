import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPredictor:
    def __init__(self, model_path):
        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        artifact = joblib.load(model_p)
        
        if isinstance(artifact, dict):
            self.model = artifact['model']
            self.features = artifact.get('features', None)
            self.metrics = artifact.get('metrics', {})
            self.model_type = artifact.get('model_type', 'unknown')
            self.training_info = artifact.get('training_info', {})
        else:
            self.model = artifact
            self.features = None
            self.metrics = {}
            self.model_type = 'unknown'
            self.training_info = {}
        
        self.explainer = shap.TreeExplainer(self.model)
        
        if self.features is None:
            if hasattr(self.model, 'feature_name_'):
                self.features = self.model.feature_name_
            elif hasattr(self.model, 'feature_names_in_'):
                self.features = self.model.feature_names_in_
        
        logging.info(f"Loaded {self.model_type} model")
        if self.metrics:
            logging.info(f"Training metrics - RMSE: {self.metrics.get('rmse', 0):.4f}, "
                        f"MAE: {self.metrics.get('mae', 0):.4f}, R²: {self.metrics.get('r2', 0):.4f}")
    
    def predict(self, input_data):
        if self.features is not None:
            missing_features = set(self.features) - set(input_data.columns)
            if missing_features:
                logging.warning(f"Missing features: {missing_features}")
                for feat in missing_features:
                    input_data[feat] = 0
            input_data = input_data[self.features]
        
        return self.model.predict(input_data)
    
    def evaluate_on_test_data(self, test_data, n_samples=20):
        """
        Evaluate model on test data - compare predictions vs actual.
        """
        logging.info("\n" + "="*80)
        logging.info("MODEL EVALUATION ON TEST DATA")
        logging.info("="*80)
        
        if test_data.empty:
            logging.error("No test data provided!")
            return
        
        # Sample data
        sample_data = test_data.sample(n=min(n_samples, len(test_data)), random_state=42)
        
        # Make predictions
        predictions = self.predict(sample_data)
        
        # Create results DataFrame
        results = sample_data[['player', 'team', 'date', 'fantasy_points']].copy()
        results['Predicted_Points'] = predictions
        results['Error'] = results['fantasy_points'] - results['Predicted_Points']
        results['Abs_Error'] = np.abs(results['Error'])
        results['Pct_Error'] = (results['Abs_Error'] / (results['fantasy_points'] + 1)) * 100
        
        # Calculate metrics
        rmse = np.sqrt((results['Error']**2).mean())
        mae = results['Abs_Error'].mean()
        r2 = 1 - (results['Error']**2).sum() / ((results['fantasy_points'] - results['fantasy_points'].mean())**2).sum()
        mape = results['Pct_Error'].mean()
        
        print("\n" + "="*80)
        print("PREDICTION VS ACTUAL COMPARISON")
        print("="*80)
        print(results[['player', 'team', 'fantasy_points', 'Predicted_Points', 'Error']].to_string(index=False))
        
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"RMSE: {rmse:.2f} points")
        print(f"MAE: {mae:.2f} points")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.1f}%")
        print("="*80)
        
        # Show best and worst predictions
        print("\n" + "="*80)
        print("BEST PREDICTIONS (smallest error):")
        print("="*80)
        best = results.nsmallest(5, 'Abs_Error')[['player', 'fantasy_points', 'Predicted_Points', 'Error']]
        print(best.to_string(index=False))
        
        print("\n" + "="*80)
        print("WORST PREDICTIONS (largest error):")
        print("="*80)
        worst = results.nlargest(5, 'Abs_Error')[['player', 'fantasy_points', 'Predicted_Points', 'Error']]
        print(worst.to_string(index=False))
        print("="*80)
        
        return results
    
    def explain_prediction(self, input_data, player_idx=0):
        """
        Explain a single prediction using SHAP.
        """
        if self.features is not None:
            input_data_filtered = input_data[self.features]
        else:
            input_data_filtered = input_data
        
        shap_values = self.explainer.shap_values(input_data_filtered.iloc[player_idx:player_idx+1])
        
        if len(shap_values.shape) > 1:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        explanation = {
            feature: shap_val 
            for feature, shap_val in zip(self.features, shap_vals)
        }
        
        return explanation
    
    def recommend_team_for_match(self, team1_players, team2_players, 
                                 budget=100, max_players=11,
                                 min_wk=1, min_bat=1, min_bowl=1, min_ar=1,
                                 max_from_one_team=7):
        """
        Recommend fantasy team for a specific match between two teams.
        """
        logging.info("\n" + "="*80)
        logging.info("GENERATING FANTASY TEAM FOR MATCH")
        logging.info("="*80)
        
        # Combine both teams
        all_players = pd.concat([team1_players, team2_players], ignore_index=True)
        
        if all_players.empty:
            logging.error("No players provided!")
            return pd.DataFrame()
        
        team1_name = team1_players['team'].iloc[0] if not team1_players.empty else 'Team1'
        team2_name = team2_players['team'].iloc[0] if not team2_players.empty else 'Team2'
        
        logging.info(f"Match: {team1_name} vs {team2_name}")
        logging.info(f"{team1_name}: {len(team1_players)} players")
        logging.info(f"{team2_name}: {len(team2_players)} players")
        
        # Load roles
        roles_path = Path('data/processed/player_roles.csv')
        if roles_path.exists():
            roles_df = pd.read_csv(roles_path)
            all_players = all_players.merge(roles_df[['player', 'role']], on='player', how='left')
        else:
            logging.error(f"Player roles not found at {roles_path}")
            return pd.DataFrame()
        
        all_players = all_players[all_players['role'].notna()].copy()
        
        if all_players.empty:
            logging.error("No players with roles!")
            return pd.DataFrame()
        
        # Make predictions
        predictions = self.predict(all_players)
        all_players['predicted_points'] = predictions
        
        # Assign costs
        if 'cost' not in all_players.columns:
            all_players['cost'] = 7.5 + (all_players['predicted_points'] / all_players['predicted_points'].quantile(0.95)) * 3.5
            all_players['cost'] = all_players['cost'].clip(7.5, 11.0).round(1)
        
        all_players['value'] = all_players['predicted_points'] / all_players['cost']
        all_players = all_players.sort_values('value', ascending=False).reset_index(drop=True)
        
        # Show available players
        logging.info(f"\nAVAILABLE PLAYERS BY ROLE:")
        for role in ['WK', 'BAT', 'BOWL', 'AR']:
            role_players = all_players[all_players['role'] == role]
            if len(role_players) > 0:
                logging.info(f"\n{role} ({len(role_players)} total):")
                logging.info(f"  {team1_name}: {len(role_players[role_players['team'] == team1_name])}")
                logging.info(f"  {team2_name}: {len(role_players[role_players['team'] == team2_name])}")
        
        # Team selection
        selected_team = []
        remaining_budget = budget
        remaining_slots = max_players
        
        role_requirements = {'WK': min_wk, 'BAT': min_bat, 'BOWL': min_bowl, 'AR': min_ar}
        role_selected = {'WK': 0, 'BAT': 0, 'BOWL': 0, 'AR': 0}
        team_counts = {team1_name: 0, team2_name: 0}
        
        logging.info(f"\nPHASE 1: Filling minimum role requirements")
        
        # Phase 1: Fill requirements
        for role, min_count in role_requirements.items():
            role_players = all_players[all_players['role'] == role].copy()
            
            for _, player in role_players.iterrows():
                if role_selected[role] >= min_count:
                    break
                
                player_team = player['team']
                
                if team_counts.get(player_team, 0) >= max_from_one_team:
                    continue
                
                if remaining_budget >= player['cost'] and remaining_slots > 0:
                    selected_team.append(player.to_dict())
                    remaining_budget -= player['cost']
                    remaining_slots -= 1
                    role_selected[role] += 1
                    team_counts[player_team] = team_counts.get(player_team, 0) + 1
                    
                    logging.info(f"  ✓ {player['player']} ({role}, {player_team}) - "
                               f"{player['predicted_points']:.1f} pts, {player['cost']} cr")
        
        # Check requirements
        unmet = [role for role, count in role_selected.items() if count < role_requirements[role]]
        
        if unmet:
            logging.error(f"\nFailed to meet requirements for: {unmet}")
            return pd.DataFrame(selected_team) if selected_team else pd.DataFrame()
        
        # Phase 2: Fill remaining
        logging.info(f"\nPHASE 2: Filling remaining {remaining_slots} slots")
        
        selected_players = [p['player'] for p in selected_team]
        
        for _, player in all_players.iterrows():
            if remaining_slots == 0:
                break
            
            if player['player'] not in selected_players:
                player_team = player['team']
                
                if team_counts.get(player_team, 0) >= max_from_one_team:
                    continue
                
                if remaining_budget >= player['cost']:
                    selected_team.append(player.to_dict())
                    remaining_budget -= player['cost']
                    remaining_slots -= 1
                    role_selected[player['role']] += 1
                    team_counts[player_team] = team_counts.get(player_team, 0) + 1
                    
                    logging.info(f"  ✓ {player['player']} ({player['role']}, {player_team}) - "
                               f"{player['predicted_points']:.1f} pts, {player['cost']} cr")
        
        team_df = pd.DataFrame(selected_team)
        
        if not team_df.empty:
            logging.info(f"\n{'='*80}")
            logging.info("TEAM SUMMARY")
            logging.info(f"{'='*80}")
            logging.info(f"Players: {len(team_df)}/{max_players}")
            logging.info(f"Budget: {budget - remaining_budget:.1f}/{budget}")
            logging.info(f"Expected points: {team_df['predicted_points'].sum():.1f}")
            logging.info(f"\nBy role: {dict(team_df['role'].value_counts())}")
            logging.info(f"By team: {dict(team_df['team'].value_counts())}")
        
        return team_df


def get_match_squads(df, team1_name, team2_name, lookback_days=180):
    """Extract recent player data for both teams."""
    team1_data = df[df['team'] == team1_name].copy()
    team2_data = df[df['team'] == team2_name].copy()
    
    if team1_data.empty or team2_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    if 'date' in df.columns:
        cutoff = df['date'].max() - timedelta(days=lookback_days)
        team1_data = team1_data[team1_data['date'] >= cutoff]
        team2_data = team2_data[team2_data['date'] >= cutoff]
    
    team1_players = team1_data.sort_values('date').groupby('player').tail(1).reset_index(drop=True)
    team2_players = team2_data.sort_values('date').groupby('player').tail(1).reset_index(drop=True)
    
    return team1_players, team2_players


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fantasy Cricket Model Predictor")
    parser.add_argument('--model_type', type=str, default='lightgbm', 
                       choices=['lightgbm', 'xgboost'])
    parser.add_argument('--team1', type=str, default=None,
                       help='Team 1 name (for match prediction mode)')
    parser.add_argument('--team2', type=str, default=None,
                       help='Team 2 name (for match prediction mode)')
    parser.add_argument('--lookback_days', type=int, default=210)
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation mode (predict vs actual)')
    parser.add_argument('--n_samples', type=int, default=30,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    MODEL_PATH = f'model_artifacts/{args.model_type}_model.joblib'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.parquet'
    TRAINING_CUTOFF_DATE = '2024-06-30'
    
    try:
        predictor = ModelPredictor(model_path=MODEL_PATH)
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        
        # Filter T20 only
        if 'match_type' in df.columns:
            df = df[df['match_type'] == 'T20'].copy()
        
        # MODE 1: Evaluation (default if no teams specified)
        if args.team1 is None or args.team2 is None or args.evaluate:
            logging.info("Running in EVALUATION mode (comparing predictions vs actual)")
            
            # Get test data (after training cutoff)
            test_data = df[df['date'] > TRAINING_CUTOFF_DATE].copy()
            
            if test_data.empty:
                logging.warning("No test data after cutoff. Using recent training data.")
                test_data = df[df['date'] <= TRAINING_CUTOFF_DATE].tail(100)
            
            # Evaluate
            results = predictor.evaluate_on_test_data(test_data, n_samples=args.n_samples)
            
            # Explain one prediction
            if not test_data.empty:
                sample = test_data.sample(1, random_state=42)
                explanation = predictor.explain_prediction(sample)
                
                player_name = sample['player'].iloc[0]
                actual = sample['fantasy_points'].iloc[0]
                predicted = predictor.predict(sample)[0]
                
                print(f"\n{'='*80}")
                print(f"DETAILED EXPLANATION FOR: {player_name}")
                print(f"{'='*80}")
                print(f"Actual: {actual:.1f} | Predicted: {predicted:.1f} | Error: {actual - predicted:.1f}")
                print(f"\nTop 10 Feature Contributions:")
                print("-"*80)
                sorted_exp = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
                for i, (feat, val) in enumerate(sorted_exp[:10], 1):
                    direction = "↑" if val > 0 else "↓"
                    print(f"{i:2d}. {feat:40s} {direction} {val:+8.4f}")
                print("="*80)
        
        # MODE 2: Match Team Selection
        else:
            logging.info(f"Running in TEAM SELECTION mode for {args.team1} vs {args.team2}")
            
            team1_players, team2_players = get_match_squads(
                df, args.team1, args.team2, args.lookback_days
            )
            
            if team1_players.empty or team2_players.empty:
                logging.error(f"\nCould not find players for both teams!")
                logging.error("\nAvailable teams:")
                print(df['team'].value_counts().head(30))
                exit(1)
            
            team = predictor.recommend_team_for_match(
                team1_players, team2_players,
                budget=100, max_players=11,
                min_wk=1, min_bat=1, min_bowl=1, min_ar=1,
                max_from_one_team=7
            )
            
            if not team.empty:
                print("\n" + "="*80)
                print("RECOMMENDED FANTASY TEAM")
                print("="*80)
                print(team[['player', 'team', 'role', 'predicted_points', 'cost']].to_string(index=False))
                print("\n" + "="*80)
                print(f"Total predicted points: {team['predicted_points'].sum():.1f}")
                print(f"Total cost: {team['cost'].sum():.1f}/100")
                print("="*80)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
