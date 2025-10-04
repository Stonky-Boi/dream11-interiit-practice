import streamlit as st
import pandas as pd
from UI.utils import solve_team_optimization
from gtts import gTTS
import io

FEATURE_NAME_MAP = {
    'roll_fantasy_points_3': 'excellent form in the last 3 matches',
    'roll_fantasy_points_5': 'consistent performance over the last 5 matches',
    'roll_fantasy_points_10': 'strong record over the last 10 matches',
    'roll_runs_scored_3': 'high run-scoring in the last 3 matches',
    'roll_runs_scored_5': 'consistent run-scoring over the last 5 matches',
    'roll_runs_scored_10': 'good history of scoring runs',
    'roll_wickets_3': 'good wicket-taking form recently',
    'roll_wickets_5': 'consistent wicket-taker',
    'roll_wickets_10': 'proven history of taking wickets',
    'venue_avg_fp': 'strong performance record at this venue'
}

def get_squad_data(team1, team2, all_data_df):
    """Fetches squad data by filtering for the two teams."""
    squad_players = all_data_df[all_data_df['team'].isin([team1, team2])]['player'].unique()
    if len(squad_players) == 0:
        return pd.DataFrame()
    latest_stats = all_data_df[all_data_df['player'].isin(squad_players)].sort_values('date').groupby('player').tail(1)
    return latest_stats

def generate_explanation_audio(player_name, explanation):
    """Generates a spoken explanation for a player's selection using gTTS."""
    sorted_explanation = sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)
    top_feature_technical = sorted_explanation[0][0]
    contribution = sorted_explanation[0][1]

    top_feature_friendly = FEATURE_NAME_MAP.get(top_feature_technical, top_feature_technical.replace('_', ' '))
    if contribution > 0:
        text = f"{player_name} is a strong pick, primarily due to their {top_feature_friendly}."
    else:
        text = f"Despite a weaker recent {top_feature_friendly}, {player_name} is still a valuable part of the team."

    # Generate audio in memory
    tts = gTTS(text, lang='en', tld='co.in') # Using Indian English accent
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def show_page(predictor, data_df, roles_df):
    """Renders the Product UI page for team recommendation."""
    st.title("🎯 Dream11 AI Team Builder")
    st.write("Select two competing teams to get an AI-powered optimal team recommendation.")

    all_teams = sorted(data_df['team'].unique())
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=all_teams.index("India") if "India" in all_teams else 0)
    with col2:
        team2 = st.selectbox("Select Team 2", all_teams, index=all_teams.index("England") if "England" in all_teams else 1)

    if st.button("🚀 Recommend Optimal Team"):
        if not team1 or not team2 or team1 == team2:
            st.error("Please select two different teams.")
        else:
            with st.spinner("Analyzing players and building your team..."):
                squad_df = get_squad_data(team1, team2, data_df)
                if squad_df.empty:
                    st.error("Could not find player data. Please check the team names.")
                else:
                    squad_df['predicted_points'] = predictor.predict(squad_df)
                    squad_df = squad_df.merge(roles_df, on='player', how='left').fillna({'role': 'BAT'})
                    
                    # **FIX: Save the calculated team to session state**
                    st.session_state.recommended_team = solve_team_optimization(squad_df, points_col='predicted_points')
                    # Reset audio player state if a new team is generated
                    if 'play_audio_for_player' in st.session_state:
                        del st.session_state.play_audio_for_player

    # **FIX: Display logic now checks session state, so it persists after button clicks**
    if 'recommended_team' in st.session_state:
        st.subheader("🎉 Your AI-Generated Dream Team")
        recommended_team = st.session_state.recommended_team
        
        for index, row in recommended_team.sort_values(by='predicted_points', ascending=False).iterrows():
            col_player, col_audio = st.columns([4, 1])
            player_info = f"**{row['player']}** ({row['team']} - {row['role']}) | **Predicted Points: {row['predicted_points']:.2f}**"
            col_player.markdown(player_info)
            
            # Use session state to manage which "Why?" button was clicked
            if col_audio.button(f"Why?", key=f"why_{index}"):
                st.session_state.play_audio_for_player = index
            
            # If this player's "Why?" button was clicked, generate and show the audio
            if st.session_state.get('play_audio_for_player') == index:
                explanation = predictor.explain_prediction(pd.DataFrame([row]))
                audio_bytes = generate_explanation_audio(row['player'], explanation)
                col_player.audio(audio_bytes, format='audio/mp3')
        
        total_points = recommended_team['predicted_points'].sum()
        st.success(f"**Total Predicted Points: {total_points:.2f}**")