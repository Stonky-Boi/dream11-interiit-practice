"""
Product UI - Dream11 Team Recommendation Interface
Interactive team builder with explainability features
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.predict_model import Dream11Predictor
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import gTTS, but make it optional
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False

class ProductUI:
    """Product UI for team recommendations"""
    
    def __init__(self):
        self.load_predictor()
        self.load_data()
    
    def load_predictor(self):
        """Load the trained model"""
        try:
            self.predictor = Dream11Predictor()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please train the model first: python model/train_model.py")
            st.stop()
    
    def load_data(self):
        """Load historical player data"""
        data_path = Path('data/processed/training_data_2024-06-30.csv')
        if not data_path.exists():
            st.error("Training data not found. Please run feature engineering first.")
            st.stop()
        
        self.player_data = pd.read_csv(data_path)
        self.player_data['date'] = pd.to_datetime(self.player_data['date'])
    
    def get_latest_player_features(self, player_name):
        """Get most recent features for a player"""
        player_matches = self.player_data[self.player_data['player'] == player_name]
        if len(player_matches) > 0:
            return player_matches.sort_values('date').iloc[-1]
        return None
    
    def get_squad_features(self, team1, team2, match_date=None):
        """Get features for players from both teams"""
        if match_date is None:
            match_date = self.player_data['date'].max()
        else:
            match_date = pd.to_datetime(match_date)
        
        # Get recent players from both teams
        recent_cutoff = match_date - pd.Timedelta(days=365)
        
        team1_players = self.player_data[
            (self.player_data['team'] == team1) & 
            (self.player_data['date'] >= recent_cutoff) &
            (self.player_data['date'] <= match_date)
        ]
        
        team2_players = self.player_data[
            (self.player_data['team'] == team2) & 
            (self.player_data['date'] >= recent_cutoff) &
            (self.player_data['date'] <= match_date)
        ]
        
        # Get latest stats for each player
        team1_latest = team1_players.groupby('player').last().reset_index()
        team2_latest = team2_players.groupby('player').last().reset_index()
        
        all_players = pd.concat([team1_latest, team2_latest], ignore_index=True)
        
        # Filter players with sufficient data
        all_players = all_players[all_players['career_matches'] >= 5]
        
        return all_players
    
    def select_dream_team(self, predictions_df):
        """Select best 11 players following Dream11 constraints"""
        # Dream11 constraints
        MIN_PLAYERS_PER_TEAM = 1
        MAX_PLAYERS_PER_TEAM = 7
        TEAM_SIZE = 11
        
        # Role constraints: 1-8 for each role category
        MIN_WK = 1
        MAX_WK = 4
        MIN_BAT = 1
        MAX_BAT = 8
        MIN_BOWL = 1
        MAX_BOWL = 8
        MIN_AR = 1
        MAX_AR = 4
        
        predictions_df = predictions_df.sort_values('predicted_fantasy_points', ascending=False).reset_index(drop=True)
        
        # Get unique teams
        teams = predictions_df['team'].unique()
        if len(teams) < 2:
            st.error("❌ Need players from both teams")
            return predictions_df.head(TEAM_SIZE)
        
        team1, team2 = teams[0], teams[1]
        
        # Separate by role
        wk_players = predictions_df[predictions_df['role'] == 'Wicket-Keeper'].head(MAX_WK)
        bat_players = predictions_df[predictions_df['role'] == 'Batsman'].head(MAX_BAT * 2)
        bowl_players = predictions_df[predictions_df['role'] == 'Bowler'].head(MAX_BOWL * 2)
        ar_players = predictions_df[predictions_df['role'] == 'All-Rounder'].head(MAX_AR * 2)
        
        # Ensure minimums are met
        if len(wk_players) < MIN_WK:
            st.error(f"❌ Need at least {MIN_WK} wicket-keeper(s), found {len(wk_players)}")
            return predictions_df.head(TEAM_SIZE)
        
        # Greedy selection with constraint checking
        selected = []
        role_counts = {'Wicket-Keeper': 0, 'Batsman': 0, 'Bowler': 0, 'All-Rounder': 0}
        team_counts = {team1: 0, team2: 0}
        
        # Priority order: Try to get at least minimum of each role first
        for idx, row in predictions_df.iterrows():
            if len(selected) >= TEAM_SIZE:
                break
            
            role = row['role']
            team = row['team']
            
            # Check team constraint
            if team_counts.get(team, 0) >= MAX_PLAYERS_PER_TEAM:
                continue
            
            # Check role constraints
            if role == 'Wicket-Keeper':
                if role_counts[role] >= MAX_WK:
                    continue
            elif role == 'Batsman':
                if role_counts[role] >= MAX_BAT:
                    continue
            elif role == 'Bowler':
                if role_counts[role] >= MAX_BOWL:
                    continue
            elif role == 'All-Rounder':
                if role_counts[role] >= MAX_AR:
                    continue
            
            # Add player
            selected.append(idx)
            role_counts[role] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
        
        # Check if we have minimum constraints met
        if len(selected) < TEAM_SIZE:
            st.warning(f"⚠️ Could only select {len(selected)} players with constraints. Need {TEAM_SIZE}.")
            # Fill remaining with best available
            remaining = predictions_df[~predictions_df.index.isin(selected)]
            for idx, row in remaining.iterrows():
                if len(selected) >= TEAM_SIZE:
                    break
                selected.append(idx)
                role_counts[row['role']] += 1
                team_counts[row['team']] = team_counts.get(row['team'], 0) + 1
        
        # Verify minimum role constraints
        if role_counts['Wicket-Keeper'] < MIN_WK:
            st.error(f"❌ Only {role_counts['Wicket-Keeper']} wicket-keeper(s). Need at least {MIN_WK}.")
        if role_counts['Batsman'] < MIN_BAT:
            st.warning(f"⚠️ Only {role_counts['Batsman']} batsman/batsmen. Recommended at least {MIN_BAT}.")
        if role_counts['Bowler'] < MIN_BOWL:
            st.warning(f"⚠️ Only {role_counts['Bowler']} bowler(s). Recommended at least {MIN_BOWL}.")
        if role_counts['All-Rounder'] < MIN_AR:
            st.warning(f"⚠️ Only {role_counts['All-Rounder']} all-rounder(s). Recommended at least {MIN_AR}.")
        
        # Verify team constraints
        for team in [team1, team2]:
            if team_counts.get(team, 0) < MIN_PLAYERS_PER_TEAM:
                st.error(f"❌ Only {team_counts.get(team, 0)} player(s) from {team}. Need at least {MIN_PLAYERS_PER_TEAM}.")
        
        dream_team = predictions_df.loc[selected[:TEAM_SIZE]].copy()
        dream_team = dream_team.sort_values('predicted_fantasy_points', ascending=False).reset_index(drop=True)
        
        return dream_team
    
    def explain_player_selection(self, player_row):
        """Generate explanation for player selection"""
        explanations = []
        
        # Recent form
        recent_avg = player_row.get('avg_fantasy_points_last_5', 0)
        if pd.notna(recent_avg):
            explanations.append(f"📊 Recent form: Averaging **{recent_avg:.1f} points** in last 5 matches")
        
        # Venue performance
        venue_avg = player_row.get('venue_avg_fantasy_points', 0)
        venue_matches = player_row.get('venue_matches', 0)
        if pd.notna(venue_avg) and venue_matches > 0:
            explanations.append(f"🏟️ Venue expertise: **{venue_avg:.1f} points** average ({int(venue_matches)} matches at venue)")
        
        # Career stats
        career_avg = player_row.get('career_avg_fantasy_points', 0)
        career_matches = player_row.get('career_matches', 0)
        if pd.notna(career_avg):
            explanations.append(f"📈 Career average: **{career_avg:.1f} points** ({int(career_matches)} matches)")
        
        # Role-specific insights
        role = player_row.get('role', 'Unknown')
        if role == 'Batsman':
            avg_runs = player_row.get('avg_runs_last_5', 0)
            avg_sr = player_row.get('avg_strike_rate_last_5', 0)
            if pd.notna(avg_runs):
                explanations.append(f"🏏 Batting: **{avg_runs:.1f} runs** per match (SR: {avg_sr:.1f})")
        elif role == 'Bowler':
            avg_wickets = player_row.get('avg_wickets_last_5', 0)
            avg_econ = player_row.get('avg_economy_last_5', 0)
            if pd.notna(avg_wickets):
                explanations.append(f"⚡ Bowling: **{avg_wickets:.2f} wickets** per match (Econ: {avg_econ:.2f})")
        elif role == 'All-Rounder':
            avg_runs = player_row.get('avg_runs_last_5', 0)
            avg_wickets = player_row.get('avg_wickets_last_5', 0)
            if pd.notna(avg_runs) and pd.notna(avg_wickets):
                explanations.append(f"🌟 All-round: **{avg_runs:.1f} runs** + **{avg_wickets:.2f} wickets** per match")
        
        # Form trend
        form_trend = player_row.get('form_trend', 0)
        if pd.notna(form_trend):
            if form_trend > 5:
                explanations.append(f"📈 **Rising form** (momentum: +{form_trend:.1f})")
            elif form_trend < -5:
                explanations.append(f"📉 Recent dip in form ({form_trend:.1f})")
        
        return explanations
    
    def create_player_card(self, player_row, rank):
        """Create visual player card"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            ### #{rank} {player_row['player']}
            **{player_row['role']}** | {player_row['team']}
            """)
        
        with col2:
            st.metric("Predicted Points", f"{player_row['predicted_fantasy_points']:.0f}")
        
        # Explanation
        with st.expander("💡 Why this player?"):
            explanations = self.explain_player_selection(player_row)
            for exp in explanations:
                st.markdown(f"- {exp}")
        
        st.divider()
    
    def generate_audio_summary(self, dream_team):
        """Generate audio summary of team selection"""
        if not GTTS_AVAILABLE:
            return None
        
        summary_text = f"Your Dream11 team has been generated. "
        summary_text += f"The predicted total fantasy points is {dream_team['predicted_fantasy_points'].sum():.0f}. "
        summary_text += f"The team includes {len(dream_team[dream_team['role']=='Batsman'])} batsmen, "
        summary_text += f"{len(dream_team[dream_team['role']=='Bowler'])} bowlers, "
        summary_text += f"{len(dream_team[dream_team['role']=='All-Rounder'])} all-rounders, "
        summary_text += f"and {len(dream_team[dream_team['role']=='Wicket-Keeper'])} wicket-keeper. "
        
        top_player = dream_team.iloc[0]
        summary_text += f"Your top pick is {top_player['player']}, predicted to score {top_player['predicted_fantasy_points']:.0f} points."
        
        try:
            # Generate audio
            tts = gTTS(text=summary_text, lang='en', slow=False)
            tts.save("team_summary.mp3")
            return summary_text
        except:
            return None
    
    def run(self):
        """Run the Product UI"""
        st.title("🏏 Dream11 Team Builder with AI")
        st.markdown("### Your Ultimate Fantasy Cricket Team Selection Tool")
        st.markdown("---")
        
        # Get unique teams from data
        all_teams = sorted(self.player_data['team'].unique())
        
        # Input section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            team1 = st.selectbox("Team 1", options=all_teams, index=0)
        
        with col2:
            team2_options = [t for t in all_teams if t != team1]
            team2 = st.selectbox("Team 2", options=team2_options, index=0 if len(team2_options) > 0 else 0)
        
        with col3:
            max_date = self.player_data['date'].max()
            match_date = st.date_input("Match Date", value=max_date, max_value=max_date)
        
        st.markdown("---")
        
        if st.button("🚀 Generate Dream Team", type="primary", use_container_width=True):
            start_time = datetime.now()
            
            with st.spinner("🔍 Analyzing players and predicting performance..."):
                # Get squad features
                squad_features = self.get_squad_features(team1, team2, match_date)
                
                if len(squad_features) < 11:
                    st.error(f"❌ Insufficient player data. Found only {len(squad_features)} players. Need at least 11.")
                    st.info("Try selecting different teams or a different date.")
                    return
                
                # Predict fantasy points
                predictions = self.predictor.predict(squad_features)
                
                # Select dream team
                dream_team = self.select_dream_team(predictions)
            
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            
            # Display results
            st.success(f"Dream Team Generated in {time_taken:.2f} seconds!")
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Points", f"{dream_team['predicted_fantasy_points'].sum():.0f}")
            with col2:
                st.metric("Batsmen", len(dream_team[dream_team['role']=='Batsman']))
            with col3:
                st.metric("Bowlers", len(dream_team[dream_team['role']=='Bowler']))
            with col4:
                st.metric("All-Rounders", len(dream_team[dream_team['role']=='All-Rounder']))
            with col5:
                st.metric("Wicket-Keepers", len(dream_team[dream_team['role']=='Wicket-Keeper']))
            
            st.markdown("---")
            
            # Audio summary
            if GTTS_AVAILABLE:
                with st.spinner("🎙️ Generating audio summary..."):
                    summary_text = self.generate_audio_summary(dream_team)
                    if summary_text:
                        try:
                            audio_file = open("team_summary.mp3", "rb")
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/mp3")
                            audio_file.close()
                        except:
                            pass
            
            # Player cards
            st.markdown("## 🌟 Your Dream Team")
            
            for idx, (_, player) in enumerate(dream_team.iterrows(), 1):
                self.create_player_card(player, idx)
            
            # Visualizations
            st.markdown("## 📊 Team Analytics")
            
            # Points distribution
            fig1 = px.bar(
                dream_team, 
                x='player', 
                y='predicted_fantasy_points',
                color='role',
                title='Predicted Fantasy Points by Player',
                labels={'predicted_fantasy_points': 'Fantasy Points', 'player': 'Player'},
                color_discrete_map={
                    'Batsman': '#FF6B6B',
                    'Bowler': '#4ECDC4',
                    'All-Rounder': '#FFD93D',
                    'Wicket-Keeper': '#95E1D3'
                }
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Role distribution pie chart
            role_dist = dream_team['role'].value_counts()
            fig2 = px.pie(
                values=role_dist.values,
                names=role_dist.index,
                title='Team Composition by Role',
                color_discrete_map={
                    'Batsman': '#FF6B6B',
                    'Bowler': '#4ECDC4',
                    'All-Rounder': '#FFD93D',
                    'Wicket-Keeper': '#95E1D3'
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Dream11 Team Builder",
        page_icon="🏏",
        layout="wide"
    )
    
    ui = ProductUI()
    ui.run()

if __name__ == '__main__':
    main()