import pandas as pd
import numpy as np
import joblib
import shap
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
        logging.info("\n" + "="*80)
        logging.info("MODEL EVALUATION ON TEST DATA")
        logging.info("="*80)
        
        if test_data.empty:
            logging.error("No test data provided!")
            return
        
        sample_data = test_data.sample(n=min(n_samples, len(test_data)), random_state=42)
        predictions = self.predict(sample_data)
        
        results = sample_data[['player', 'team', 'date', 'fantasy_points']].copy()
        results['Predicted_Points'] = predictions
        results['Error'] = results['fantasy_points'] - results['Predicted_Points']
        results['Abs_Error'] = np.abs(results['Error'])
        results['Pct_Error'] = (results['Abs_Error'] / (results['fantasy_points'] + 1)) * 100
        
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
        
        def generate_player_rationale(player, role):
            parts = []
            
            # Recent form with stats
            if player['recent_form'] > 0 and player['avg_last_10'] > 0:
                form_pct = ((player['recent_form'] - player['avg_last_10']) / player['avg_last_10']) * 100
                if form_pct > 15:
                    parts.append(f"🔥 Hot Form: L5 avg {player['recent_form']:.1f} vs L10 avg {player['avg_last_10']:.1f} (+{form_pct:.0f}%)")
                elif form_pct > 5:
                    parts.append(f"📈 Improving: L5 avg {player['recent_form']:.1f} vs L10 avg {player['avg_last_10']:.1f} (+{form_pct:.0f}%)")
                elif form_pct < -15:
                    parts.append(f"📉 Declining: L5 avg {player['recent_form']:.1f} vs L10 avg {player['avg_last_10']:.1f} ({form_pct:.0f}%)")
                else:
                    parts.append(f"✓ Stable: L5 avg {player['recent_form']:.1f}")
            
            # Consistency
            if player['consistency'] > 0:
                cons_pct = min(player['consistency'] * 100, 100)
                if cons_pct > 70:
                    parts.append(f"💎 Consistency: {cons_pct:.0f}%")
                elif cons_pct > 50:
                    parts.append(f"✓ Consistency: {cons_pct:.0f}%")
            
            # Role-specific
            if role in ['BAT', 'AR'] and 'strike_rate' in player and player['strike_rate'] > 0:
                if player['strike_rate'] > 145:
                    parts.append(f"⚡ Explosive: SR {player['strike_rate']:.0f}")
                elif player['strike_rate'] > 125:
                    parts.append(f"🎯 Aggressive: SR {player['strike_rate']:.0f}")
            
            if role in ['BOWL', 'AR'] and 'economy_rate' in player and player['economy_rate'] > 0:
                if player['economy_rate'] < 7:
                    parts.append(f"🛡️ Economical: ER {player['economy_rate']:.1f}")
                if 'roll_wickets_5' in player and player['roll_wickets_5'] > 0.8:
                    parts.append(f"🎯 Wicket-taker: {player['roll_wickets_5']:.1f} wkts/match")
            
            return " • ".join(parts) if parts else f"Predicted: {player['predicted_points']:.1f} pts"
        
        print("\n")
        print("┌" + "─"*98 + "┐")
        print("│" + " "*20 + "DREAM11 T20 FANTASY TEAM BUILDER" + " "*46 + "│")
        print("└" + "─"*98 + "┘")
        print()
        
        all_players = pd.concat([team1_players, team2_players], ignore_index=True)
        
        if all_players.empty:
            print("❌ Error: No players found!")
            return pd.DataFrame()
        
        team1_name = team1_players['team'].iloc[0] if not team1_players.empty else 'Team1'
        team2_name = team2_players['team'].iloc[0] if not team2_players.empty else 'Team2'
        
        print(f"📋 MATCH: {team1_name} vs {team2_name}")
        print(f"👥 Available Players: {team1_name} ({len(team1_players)}) | {team2_name} ({len(team2_players)})")
        print("─"*100)
        print()
        
        # Load roles
        roles_path = Path('data/processed/player_roles.csv')
        if roles_path.exists():
            roles_df = pd.read_csv(roles_path)
            all_players = all_players.merge(roles_df[['player', 'role']], on='player', how='left')
        else:
            print("❌ Error: Player roles file not found!")
            return pd.DataFrame()
        
        all_players = all_players[all_players['role'].notna()].copy()
        
        if all_players.empty:
            print("❌ Error: No players with assigned roles!")
            return pd.DataFrame()
        
        # Predictions
        predictions = self.predict(all_players)
        all_players['predicted_points'] = predictions
        
        if 'cost' not in all_players.columns:
            raw_cost = 6.5 + (all_players['predicted_points'] / all_players['predicted_points'].quantile(0.95)) * 2.5
            raw_cost = raw_cost.clip(6.5, 9.0)
            
            def smart_round(cost):
                
                return round(cost * 2) / 2
            
            all_players['cost'] = raw_cost.apply(smart_round)

        # Calculate value (points per credit) right after cost
        all_players['value'] = all_players['predicted_points'] / all_players['cost']

        
        # Extract features
        feature_mapping = {
            'recent_form': 'weighted_fp_5',
            'avg_last_5': 'roll_fantasy_points_5',
            'avg_last_10': 'roll_fantasy_points_10',
            'consistency': 'consistency_score',
            'strike_rate': 'strike_rate',
            'economy_rate': 'economy_rate',
            'roll_wickets_5': 'roll_wickets_5',
            'roll_runs_scored_5': 'roll_runs_scored_5'
        }
        
        for key, col in feature_mapping.items():
            all_players[key] = all_players[col] if col in all_players.columns else 0
        
        all_players = all_players.sort_values('predicted_points', ascending=False).reset_index(drop=True)
        
        print("📊 TOP PERFORMERS BY ROLE")
        print("="*100)
        
        for role in ['WK', 'BAT', 'AR', 'BOWL']:
            role_players = all_players[all_players['role'] == role].head(10)
            
            if len(role_players) == 0:
                continue
            
            role_names = {'WK': 'Wicket-Keeper', 'BAT': 'Batsman', 'AR': 'All-Rounder', 'BOWL': 'Bowler'}
            print(f"\n[{role}] {role_names[role]}")
            print("─"*100)
            
            for idx, p in role_players.iterrows():
                form_emoji = ""
                if p['recent_form'] > 0 and p['avg_last_10'] > 0:
                    form_change = ((p['recent_form'] - p['avg_last_10']) / p['avg_last_10']) * 100
                    if form_change > 15:
                        form_emoji = "🔥"
                    elif form_change > 5:
                        form_emoji = "📈"
                    elif form_change < -15:
                        form_emoji = "📉"
                
                print(f"   {p['player']:<28} ({p['team']:<15}) │ {p['predicted_points']:>5.1f} pts │ {p['cost']:>4.1f} cr │ {form_emoji}")
        
        print("\n" + "="*100 + "\n")
        
        print("🎯 BUILDING YOUR DREAM TEAM")
        print("="*100)
        print(f"💰 Budget: {budget} credits │ 👥 Players: {max_players} │ ⚖️ Max from one team: {max_from_one_team}")
        print(f"📋 Requirements: {min_wk} WK, {min_bat} BAT, {min_ar} AR, {min_bowl} BOWL")
        print("="*100)
        print()
        
        selected_team = []
        remaining_budget = budget
        remaining_slots = max_players
        role_requirements = {'WK': min_wk, 'BAT': min_bat, 'BOWL': min_bowl, 'AR': min_ar}
        role_selected = {'WK': 0, 'BAT': 0, 'BOWL': 0, 'AR': 0}
        team_counts = {team1_name: 0, team2_name: 0}
        selection_log = []
        rejection_log = []
        
        all_players_sorted = all_players.sort_values('value', ascending=False).reset_index(drop=True)
        
        print("STEP 1: Filling Required Roles")
        print("─"*100)
        
        # Phase 1: Fill requirements
        for role, min_count in role_requirements.items():
            role_players = all_players_sorted[all_players_sorted['role'] == role].copy()
            
            for _, player in role_players.iterrows():
                if role_selected[role] >= min_count:
                    break
                
                player_team = player['team']
                
                if team_counts.get(player_team, 0) >= max_from_one_team:
                    rejection_log.append({'player': player['player'], 'reason': 'Team quota full', 
                                        'predicted_points': player['predicted_points'], 'cost': player['cost']})
                    continue
                
                if remaining_budget < player['cost']:
                    rejection_log.append({'player': player['player'], 'reason': f'Over budget (need {player["cost"]:.1f}cr)', 
                                        'predicted_points': player['predicted_points'], 'cost': player['cost']})
                    continue
                
                if remaining_slots <= 0:
                    break
                
                rationale = generate_player_rationale(player, role)
                
                selected_team.append(player.to_dict())
                remaining_budget -= player['cost']
                remaining_slots -= 1
                role_selected[role] += 1
                team_counts[player_team] = team_counts.get(player_team, 0) + 1
                
                print(f"\n✓ [{role}] {player['player']:<25} │ {player_team:<12} │ {player['predicted_points']:>5.1f} pts │ {player['cost']:>4.1f} cr")
                print(f"    {rationale}")
                
                selection_log.append({
                    'player': player['player'],
                    'team': player_team,
                    'role': role,
                    'predicted_points': player['predicted_points'],
                    'cost': player['cost'],
                    'rationale': rationale
                })
        
        unmet = [role for role, count in role_selected.items() if count < role_requirements[role]]
        if unmet:
            print(f"\n❌ Cannot satisfy requirements for: {', '.join(unmet)}")
            return pd.DataFrame()
        
        # Phase 2: Optimize
        if remaining_slots > 0:
            print(f"\n\nSTEP 2: Optimizing Remaining {remaining_slots} Slots")
            print("─"*100)
            
            selected_players = [p['player'] for p in selected_team]
            
            for _, player in all_players_sorted.iterrows():
                if remaining_slots == 0:
                    break
                
                if player['player'] in selected_players:
                    continue
                
                player_team = player['team']
                
                if team_counts.get(player_team, 0) >= max_from_one_team:
                    rejection_log.append({'player': player['player'], 'reason': 'Team quota full',
                                        'predicted_points': player['predicted_points'], 'cost': player['cost']})
                    continue
                
                if remaining_budget < player['cost']:
                    rejection_log.append({'player': player['player'], 'reason': f'Over budget (need {player["cost"]:.1f}cr)',
                                        'predicted_points': player['predicted_points'], 'cost': player['cost']})
                    continue
                
                rationale = generate_player_rationale(player, player['role'])
                
                selected_team.append(player.to_dict())
                remaining_budget -= player['cost']
                remaining_slots -= 1
                role_selected[player['role']] += 1
                team_counts[player_team] = team_counts.get(player_team, 0) + 1
                
                print(f"\n✓ [{player['role']}] {player['player']:<25} │ {player_team:<12} │ {player['predicted_points']:>5.1f} pts │ {player['cost']:>4.1f} cr")
                print(f"    {rationale}")
                
                selection_log.append({
                    'player': player['player'],
                    'team': player_team,
                    'role': player['role'],
                    'predicted_points': player['predicted_points'],
                    'cost': player['cost'],
                    'rationale': rationale
                })
        
        team_df = pd.DataFrame(selected_team)
        
        if team_df.empty:
            print("\n❌ Team selection failed!")
            return team_df
        
        # ===== FINAL TEAM =====
        print("\n\n")
        print("┌" + "─"*98 + "┐")
        print("│" + " "*35 + "YOUR DREAM TEAM" + " "*48 + "│")
        print("└" + "─"*98 + "┘")
        print()
        
        for idx, log in enumerate(selection_log, 1):
            role_emoji = {'WK': '🧤', 'BAT': '🏏', 'AR': '⭐', 'BOWL': '🎯'}
            print(f"{idx:>2}. {role_emoji.get(log['role'], '')} {log['player']:<25} │ {log['team']:<12} │ {log['predicted_points']:>5.1f} pts │ {log['cost']:>4.1f} cr")
        
        print("\n" + "─"*100)
        print("📊 TEAM SUMMARY")
        print("─"*100)
        print(f"👥 Players: {len(team_df)}/{max_players}")
        print(f"💰 Budget Used: {budget - remaining_budget:.1f}/{budget} cr ({(budget-remaining_budget)/budget*100:.1f}%)")
        print(f"⭐ Expected Points: {team_df['predicted_points'].sum():.1f}")
        print(f"📈 Average per Player: {team_df['predicted_points'].mean():.1f} pts")
        
        print(f"\n🎯 Role Split: ", end="")
        print(" | ".join([f"{role}: {count}" for role, count in sorted(team_df['role'].value_counts().items())]))
        
        print(f"⚖️ Team Split: ", end="")
        for team, count in sorted(team_df['team'].value_counts().items()):
            print(f"{team}: {count} ({count/len(team_df)*100:.0f}%)", end=" | ")
        print()
        
        print("\n👑 CAPTAIN RECOMMENDATIONS")
        print("─"*100)
        top_3 = team_df.nlargest(3, 'predicted_points')
        for idx, (_, row) in enumerate(top_3.iterrows(), 1):
            badge = "👑 Captain" if idx == 1 else ("🥈 Vice-Captain" if idx == 2 else "🥉 Alternative")
            print(f"{badge:20s}: {row['player']:<25} │ {row['predicted_points']:>5.1f} pts")
        
        if rejection_log:
            print("\n❌ NOTABLE EXCLUSIONS")
            print("─"*100)
            rejection_df = pd.DataFrame(rejection_log).drop_duplicates(subset=['player'])
            for _, rej in rejection_df.nlargest(5, 'predicted_points').iterrows():
                print(f"   {rej['player']:<25} │ {rej['predicted_points']:>5.1f} pts │ {rej['cost']:>4.1f} cr │ {rej['reason']}")
        
        print("\n" + "="*100)
        print("✅ Team selection complete!")
        print("="*100 + "\n")
        
        return team_df





def get_match_squads(df, team1_name, team2_name, gender='male', lookback_days=180):
    """Extract recent player data for both teams WITH GENDER FILTER."""
    # Filter by gender first
    if 'gender' in df.columns:
        df = df[df['gender'] == gender].copy()
        logging.info(f"Filtering for {gender} cricket")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    team1_data = df[df['team'] == team1_name].copy()
    team2_data = df[df['team'] == team2_name].copy()
    
    if team1_data.empty or team2_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter by lookback period
    if 'date' in df.columns:
        cutoff = df['date'].max() - timedelta(days=lookback_days)
        team1_data = team1_data[team1_data['date'] >= cutoff]
        team2_data = team2_data[team2_data['date'] >= cutoff]
    
    team1_players = team1_data.sort_values('date').groupby('player').tail(1).reset_index(drop=True)
    team2_players = team2_data.sort_values('date').groupby('player').tail(1).reset_index(drop=True)
    
    logging.info(f"{team1_name}: {len(team1_players)} players")
    logging.info(f"{team2_name}: {len(team2_players)} players")
    
    return team1_players, team2_players


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fantasy Cricket Model Predictor")
    parser.add_argument('--model_type', type=str, default='lightgbm', 
                       choices=['lightgbm', 'xgboost'])
    parser.add_argument('--team1', type=str, default=None,
                       help='Team 1 name (for match prediction mode)')
    parser.add_argument('--team2', type=str, default=None,
                       help='Team 2 name (for match prediction mode)')
    parser.add_argument('--gender', type=str, default='male',
                       choices=['male', 'female'],
                       help='Cricket gender (male or female)')
    parser.add_argument('--lookback_days', type=int, default=180,
                       help='Consider players active in last N days')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation mode (predict vs actual)')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    MODEL_PATH = f'model_artifacts/{args.model_type}_{args.gender}_model.joblib'
    PROCESSED_DATA_PATH = 'data/processed/final_model_data.csv'
    TRAINING_CUTOFF_DATE = '2024-06-30'
    
    try:
        predictor = ModelPredictor(model_path=MODEL_PATH)
        
        # Load CSV with date parsing
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['date'])
        
        # Filter T20 only
        if 'match_type' in df.columns:
            df = df[df['match_type'] == 'T20'].copy()
        
        # MODE 1: Evaluation (default if no teams specified)
        if args.team1 is None or args.team2 is None or args.evaluate:
            logging.info("Running in EVALUATION mode (comparing predictions vs actual)")
            
            test_data = df[df['date'] > TRAINING_CUTOFF_DATE].copy()
            
            if test_data.empty:
                logging.warning("No test data after cutoff. Using recent training data.")
                test_data = df[df['date'] <= TRAINING_CUTOFF_DATE].tail(100)
            
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
                df, args.team1, args.team2, 
                gender=args.gender, 
                lookback_days=args.lookback_days
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
