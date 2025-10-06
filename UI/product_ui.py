import streamlit as st
import pandas as pd
from UI.utils import solve_team_optimization
from gtts import gTTS
import io

FEATURE_NAME_MAP = {
    'roll_runs_scored_3': 'strong run-scoring form in the last 3 matches',
    'roll_runs_scored_5': 'consistent run-scoring over the last 5 matches',
    'roll_runs_scored_10': 'proven long-term record of scoring runs',
    'roll_balls_faced_3': 'good record of batting for long periods recently',
    'roll_balls_faced_5': 'spending significant time at the crease in recent matches',
    'roll_balls_faced_10': 'proven ability to build a long innings',
    'roll_wickets_3': 'excellent recent wicket-taking form',
    'roll_wickets_5': 'being a consistent wicket-taker over the last 5 matches',
    'roll_wickets_10': 'proven history as a long-term wicket-taker',
    'roll_runs_conceded_3': 'excellent recent economy rate',
    'roll_runs_conceded_5': 'solid and consistent economy rate',
    'roll_runs_conceded_10': 'reliable and economical bowling history',
    'roll_catches_3': 'great recent fielding with multiple catches',
    'roll_catches_5': 'safe pair of hands in the field recently',
    'roll_catches_10': 'reliable and consistent fielding'
}

def get_squad_data(team1, team2, all_data_df):
    squad_players = all_data_df[all_data_df['team'].isin([team1, team2])]['player'].unique()
    if len(squad_players) == 0: return pd.DataFrame()
    latest_stats = all_data_df[all_data_df['player'].isin(squad_players)].sort_values('date').groupby('player').tail(1)
    return latest_stats

def generate_explanation_audio(player_name, explanation):
    sorted_explanation = sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)
    top_feature_technical = sorted_explanation[0][0]
    contribution = sorted_explanation[0][1]
    top_feature_friendly = FEATURE_NAME_MAP.get(top_feature_technical, top_feature_technical.replace('_', ' '))
    if contribution > 0:
        text = f"{player_name} is a strong pick, primarily due to their {top_feature_friendly}."
    else:
        text = f"Despite a weaker {top_feature_friendly}, {player_name} is still a valuable part of the team."
    tts = gTTS(text, lang='en', tld='co.in')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def show_page(predictor, data_df, roles_df):
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
                    st.session_state.recommended_team = solve_team_optimization(squad_df, points_col='predicted_points')
                    if 'play_audio_for_player' in st.session_state:
                        del st.session_state.play_audio_for_player
    
    if 'recommended_team' in st.session_state:
        st.subheader("🎉 Your AI-Generated Dream Team")
        recommended_team = st.session_state.recommended_team
        for index, row in recommended_team.sort_values(by='predicted_points', ascending=False).iterrows():
            col_player, col_audio = st.columns([4, 1])
            player_info = f"**{row['player']}** ({row['team']} - {row['role']}) | **Predicted Points: {row['predicted_points']:.2f}**"
            col_player.markdown(player_info)
            if col_audio.button(f"Why?", key=f"why_{index}"):
                st.session_state.play_audio_for_player = index
            if st.session_state.get('play_audio_for_player') == index:
                explanation = predictor.explain_prediction(pd.DataFrame([row]))
                audio_bytes = generate_explanation_audio(row['player'], explanation)
                col_player.audio(audio_bytes, format='audio/mp3')
        total_points = recommended_team['predicted_points'].sum()
        st.success(f"**Total Predicted Points: {total_points:.2f}**")