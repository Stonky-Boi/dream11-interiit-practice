import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.predict_model import Dream11Predictor
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Try to import gTTS
try:
    from gtts import gTTS
    import tempfile
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False

class ProductUI:
    # Competition constraints
    TRAINING_CUTOFF = date(2024, 6, 30)
    PREDICTION_START = date(2024, 7, 1)
    
    def __init__(self):
        self.load_predictor()
        self.load_player_database()
    
    def load_predictor(self):
        try:
            self.predictor = Dream11Predictor()
            st.sidebar.success(f"✅ Model loaded with {len(self.predictor.feature_cols)} features")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please train the model first: python model/train_model.py")
            st.stop()
    
    def load_player_database(self):
        training_data_path = Path('data/processed/training_data_2024-06-30.csv')
        
        if not training_data_path.exists():
            st.error("Training data not found. Please run feature engineering first.")
            st.stop()
        
        self.player_data = pd.read_csv(training_data_path)
        self.player_data['date'] = pd.to_datetime(self.player_data['date'])
        
        # Get unique teams and players
        self.teams = sorted(self.player_data['team'].unique())
        self.all_players = sorted(self.player_data['player'].unique())
        
        # Get player statistics for display
        self.player_stats = self.player_data.groupby('player').agg({
            'fantasy_points': 'mean',
            'role': 'first',
            'team': 'first'
        }).reset_index()
    
    def get_team_players(self, team_name):
        team_players = self.player_data[self.player_data['team'] == team_name]['player'].unique()
        return sorted(team_players)
    
    def select_dream_team(self, predictions_df):
        MIN_PLAYERS_PER_TEAM = 1
        MAX_PLAYERS_PER_TEAM = 7
        TEAM_SIZE = 11
        MIN_WK = 1
        MAX_WK = 4
        MIN_BAT = 1
        MAX_BAT = 8
        MIN_BOWL = 1
        MAX_BOWL = 8
        MIN_AR = 1
        MAX_AR = 4
        
        predictions_df = predictions_df.sort_values('predicted_fantasy_points', ascending=False).reset_index(drop=True)
        
        teams = predictions_df['team'].unique() if 'team' in predictions_df.columns else []
        
        if len(teams) < 2:
            st.warning("Need players from both teams. Using top 11 players.")
            return predictions_df.head(TEAM_SIZE)
        
        team1, team2 = teams[0], teams[1]
        
        # Greedy selection with constraints
        selected = []
        role_counts = {'Wicket-Keeper': 0, 'Batsman': 0, 'Bowler': 0, 'All-Rounder': 0}
        team_counts = {team1: 0, team2: 0}
        
        for idx, row in predictions_df.iterrows():
            if len(selected) >= TEAM_SIZE:
                break
            
            role = row.get('role', 'All-Rounder')
            team = row.get('team', team1)
            
            # Check team constraint
            if team_counts.get(team, 0) >= MAX_PLAYERS_PER_TEAM:
                continue
            
            # Check role constraints
            if role == 'Wicket-Keeper' and role_counts[role] >= MAX_WK:
                continue
            elif role == 'Batsman' and role_counts[role] >= MAX_BAT:
                continue
            elif role == 'Bowler' and role_counts[role] >= MAX_BOWL:
                continue
            elif role == 'All-Rounder' and role_counts[role] >= MAX_AR:
                continue
            
            # Add player
            selected.append(idx)
            role_counts[role] += 1
            team_counts[team] = team_counts.get(team, 0) + 1
        
        # Fill remaining spots if needed
        if len(selected) < TEAM_SIZE:
            remaining = predictions_df[~predictions_df.index.isin(selected)]
            for idx, row in remaining.iterrows():
                if len(selected) >= TEAM_SIZE:
                    break
                selected.append(idx)
        
        dream_team = predictions_df.loc[selected[:TEAM_SIZE]].copy()
        dream_team = dream_team.sort_values('predicted_fantasy_points', ascending=False).reset_index(drop=True)
        
        return dream_team
    
    def generate_audio_summary(self, dream_team):
        if not GTTS_AVAILABLE:
            return None
        
        try:
            summary = f"Your Dream 11 team has {len(dream_team)} players. "
            
            total_predicted = dream_team['predicted_fantasy_points'].sum()
            summary += f"Expected total points: {total_predicted:.0f}. "
            
            top_3 = dream_team.head(3)
            summary += "Top 3 players are: "
            for idx, row in top_3.iterrows():
                summary += f"{row['player']}, {row.get('role', 'Player')}, {row['predicted_fantasy_points']:.0f} points. "
            
            # Generate audio
            tts = gTTS(text=summary, lang='en', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                return fp.name
        except Exception as e:
            st.warning(f"Could not generate audio: {str(e)}")
            return None
    
    def visualize_team(self, dream_team):        
        # Points distribution by role
        fig_role = px.bar(
            dream_team.groupby('role')['predicted_fantasy_points'].sum().reset_index(),
            x='role',
            y='predicted_fantasy_points',
            title='Expected Points by Role',
            labels={'predicted_fantasy_points': 'Total Points', 'role': 'Role'},
            color='role'
        )
        st.plotly_chart(fig_role, use_container_width=True)
        
        # Individual player contributions
        fig_players = go.Figure(data=[
            go.Bar(
                x=dream_team['player'],
                y=dream_team['predicted_fantasy_points'],
                marker_color=dream_team['predicted_fantasy_points'],
                marker_colorscale='Viridis',
                text=dream_team['predicted_fantasy_points'].round(1),
                textposition='auto'
            )
        ])
        fig_players.update_layout(
            title='Individual Player Predictions',
            xaxis_title='Player',
            yaxis_title='Predicted Points',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_players, use_container_width=True)
    
    def run(self):
        st.title("🏏 Dream11 Team Builder with AI")
        
        # Competition constraint warning
        st.warning(f"""
        **Competition Constraints:**
        - Training data: Up to {self.TRAINING_CUTOFF.strftime('%Y-%m-%d')}
        - Predictions: Only for matches on or after {self.PREDICTION_START.strftime('%Y-%m-%d')}
        """)
        
        st.markdown("---")
        
        # Sidebar inputs
        st.sidebar.header("⚙️ Match Configuration")
        
        match_type = st.sidebar.selectbox(
            "Match Format",
            ["T20", "ODI"],
            help="Select the match format"
        )
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            team1 = st.selectbox("Team 1", self.teams, key='team1')
        
        with col2:
            team2 = st.selectbox("Team 2", [t for t in self.teams if t != team1], key='team2')
        
        # Match date with validation
        match_date = st.sidebar.date_input(
            "Match Date",
            value=self.PREDICTION_START,
            min_value=self.PREDICTION_START,
            help=f"Must be on or after {self.PREDICTION_START.strftime('%Y-%m-%d')}"
        )
        
        # Validate match date
        if match_date < self.PREDICTION_START:
            st.sidebar.error(f"⚠️ Match date must be ≥ {self.PREDICTION_START.strftime('%Y-%m-%d')}")
            st.error(f"Invalid match date. Must be on or after {self.PREDICTION_START.strftime('%Y-%m-%d')}")
            return
        else:
            st.sidebar.success(f"✅ Valid prediction date")
        
        # Player selection
        st.sidebar.markdown("---")
        st.sidebar.header("👥 Squad Selection")
        
        squad_selection_mode = st.sidebar.radio(
            "Squad Selection Mode",
            ["Auto-select (Top players)", "Manual selection"],
            help="Auto-select uses recent form. Manual allows custom squad."
        )
        
        if squad_selection_mode == "Manual selection":
            team1_players = st.sidebar.multiselect(
                f"{team1} Players",
                self.get_team_players(team1),
                default=list(self.get_team_players(team1)[:11]),
                help=f"Select players from {team1}"
            )
            
            team2_players = st.sidebar.multiselect(
                f"{team2} Players",
                self.get_team_players(team2),
                default=list(self.get_team_players(team2)[:11]),
                help=f"Select players from {team2}"
            )
            
            squad_players = team1_players + team2_players
        else:
            # Auto-select top players based on recent form
            team1_top = self.player_stats[
                self.player_stats['team'] == team1
            ].nlargest(11, 'fantasy_points')['player'].tolist()
            
            team2_top = self.player_stats[
                self.player_stats['team'] == team2
            ].nlargest(11, 'fantasy_points')['player'].tolist()
            
            team1_players = team1_top
            team2_players = team2_top
            squad_players = team1_players + team2_players
            
            st.sidebar.info(f"Auto-selected {len(team1_players)} from {team1} and {len(team2_players)} from {team2}")
        
        st.sidebar.markdown(f"**Squad Size:** {len(squad_players)} players")
        
        # Validate squad
        if len(squad_players) < 11:
            st.sidebar.error("⚠️ Squad must have at least 11 players!")
        
        # Generate team button
        if st.sidebar.button("🎯 Generate Dream Team", type="primary", use_container_width=True, disabled=len(squad_players) < 11):
            
            with st.spinner("🔮 Analyzing players and predicting performance..."):
                try:
                    # Predict for all squad players
                    predictions = self.predictor.predict_for_squad(
                        squad_players,
                        match_type=match_type.lower(),
                        venue='',
                        team1=team1,
                        team2=team2
                    )
                    
                    if len(predictions) == 0:
                        st.error("Could not generate predictions. Players may not have sufficient historical data.")
                        st.info("Try selecting different players or check data availability.")
                        return
                    
                    # Add team information
                    predictions['team'] = predictions['player'].apply(
                        lambda p: team1 if p in team1_players else team2
                    )
                    
                    # Select dream team
                    dream_team = self.select_dream_team(predictions)
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    return
            
            st.success("✅ Dream Team Generated!")
            
            # Display results
            st.markdown("## 🏆 Your Dream 11 Team")
            st.markdown(f"**Match:** {team1} vs {team2} | **Date:** {match_date.strftime('%Y-%m-%d')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predicted Points", f"{dream_team['predicted_fantasy_points'].sum():.0f}")
            
            with col2:
                st.metric("Average Points per Player", f"{dream_team['predicted_fantasy_points'].mean():.1f}")
            
            with col3:
                st.metric("Top Scorer", dream_team.iloc[0]['player'])
            
            # Team composition
            st.markdown("### 📋 Team Composition")
            
            role_counts = dream_team['role'].value_counts()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🧤 Wicket-Keepers", role_counts.get('Wicket-Keeper', 0))
            with col2:
                st.metric("🏏 Batsmen", role_counts.get('Batsman', 0))
            with col3:
                st.metric("🔄 All-Rounders", role_counts.get('All-Rounder', 0))
            with col4:
                st.metric("⚡ Bowlers", role_counts.get('Bowler', 0))
            
            # Display team table
            st.markdown("### 🌟 Selected Players")
            
            display_df = dream_team[['player', 'role', 'team', 'predicted_fantasy_points']].copy()
            display_df.columns = ['Player', 'Role', 'Team', 'Predicted Points']
            display_df['Rank'] = range(1, len(display_df) + 1)
            display_df = display_df[['Rank', 'Player', 'Role', 'Team', 'Predicted Points']]
            display_df['Predicted Points'] = display_df['Predicted Points'].round(1)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Visualizations
            st.markdown("### 📊 Team Analytics")
            self.visualize_team(dream_team)
            
            # Player insights
            with st.expander("🔍 Detailed Player Insights"):
                for idx, row in dream_team.iterrows():
                    st.markdown(f"**{row['player']}** ({row['role']}) - {row['team']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"📊 Predicted: **{row['predicted_fantasy_points']:.1f} pts**")
                    with col2:
                        recent_form = row.get('avg_fantasy_points_last_5', 0)
                        st.write(f"📈 Last 5 matches: **{recent_form:.1f} pts**")
                    with col3:
                        career_avg = row.get('career_batting_avg', 0)
                        st.write(f"🏆 Career batting avg: **{career_avg:.1f}**")
                    
                    st.markdown("---")
            
            # Audio summary
            if GTTS_AVAILABLE:
                st.markdown("### 🔊 Audio Summary")
                with st.spinner("Generating audio..."):
                    audio_file = self.generate_audio_summary(dream_team)
                    if audio_file:
                        st.audio(audio_file)
            
            # Download options
            st.markdown("### 💾 Export Options")
            
            csv = dream_team.to_csv(index=False)
            st.download_button(
                label="📥 Download Team as CSV",
                data=csv,
                file_name=f"dream11_team_{team1}_vs_{team2}_{match_date.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Information section
        with st.expander("ℹ️ How It Works"):
            st.markdown(f"""
            ### Competition Constraints
            - **Training data cutoff:** {self.TRAINING_CUTOFF.strftime('%Y-%m-%d')}
            - **Prediction start date:** {self.PREDICTION_START.strftime('%Y-%m-%d')}
            - **No data leakage:** All features use historical data only
            
            ### Feature Set
            
            **Historical Performance:**
            - Recent form (last 3, 5, 10 matches)
            - Rolling averages for runs, wickets, fantasy points
            - EMA and form trend indicators
            
            **Career Aggregate Stats:**
            - Career statistics from Cricsheet
            - Batting: Runs, Average, Strike Rate, Hundreds, Fifties
            - Bowling: Wickets, Average, Economy, Best Figures
            - Fielding: Catches, Stumpings, Run Outs
            
            **Machine Learning Models:**
            - XGBoost, LightGBM, CatBoost ensemble
            - Trained on player-innings records
            - Weighted ensemble based on validation performance
            
            ### Dream11 Constraints
            - 11 players total
            - 1-7 players from each team
            - 1-4 Wicket-Keepers (min 1)
            - 1-8 Batsmen (min 1)
            - 1-4 All-Rounders (min 1)
            - 1-8 Bowlers (min 1)
            """)

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