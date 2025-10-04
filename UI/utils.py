import pulp as pl

def solve_team_optimization(players_df, points_col='predicted_points'):
    """
    Uses PuLP to select the optimal 11-player team based on points
    and Dream11 constraints. Can be used for predicted or actual points.
    """
    players = players_df.to_dict('index')
    player_ids = list(players.keys())

    prob = pl.LpProblem("Dream11_Team_Optimization", pl.LpMaximize)
    player_vars = pl.LpVariable.dicts("Player", player_ids, 0, 1, pl.LpBinary)

    # Objective Function: Maximize total points (either predicted or actual)
    prob += pl.lpSum([players[i][points_col] * player_vars[i] for i in player_ids])

    # Constraint 1: Total players must be 11
    prob += pl.lpSum([player_vars[i] for i in player_ids]) == 11

    # Constraint 2: Player role constraints
    roles = ['WK', 'BAT', 'AR', 'BOWL']
    for role in roles:
        prob += pl.lpSum([player_vars[i] for i in player_ids if players[i]['role'] == role]) >= 1
        prob += pl.lpSum([player_vars[i] for i in player_ids if players[i]['role'] == role]) <= 8
    
    # Constraint 3: Team constraints (assumes two teams are present in the df)
    teams = players_df['team'].unique()
    if len(teams) == 2:
        prob += pl.lpSum([player_vars[i] for i in player_ids if players[i]['team'] == teams[0]]) >= 1
        prob += pl.lpSum([player_vars[i] for i in player_ids if players[i]['team'] == teams[1]]) >= 1

    prob.solve(pl.PULP_CBC_CMD(msg=0))
    selected_player_indices = [i for i in player_ids if player_vars[i].varValue == 1]
    return players_df.loc[selected_player_indices]