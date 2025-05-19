import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (playergamelogs, 
                                     teamgamelogs, teamestimatedmetrics, playerestimatedmetrics)
import numpy as np
from typing import List
import os

def create_playergamelogs_df() -> pd.DataFrame:
    df = pd.DataFrame()

    for i in range(17, 25):
        year_string = f"20{i}-{i+1}"
    
        p_df = playergamelogs.PlayerGameLogs(season_nullable=year_string).get_data_frames()[0]
        df = pd.concat([df, p_df])

    return df

def create_teamgamelogs_df() -> pd.DataFrame:
    df = pd.DataFrame()

    for i in range(17, 25):
        year_string = f"20{i}-{i+1}"
        t_df = teamgamelogs.TeamGameLogs(season_nullable=year_string).get_data_frames()[0]
        df = pd.concat([df, t_df])
    
    return df

def create_estimatedteammetrics_df() -> pd.DataFrame:
    df = pd.DataFrame()

    for i in range(17, 25):
        year_string = f"20{i}-{i+1}"
        t_df = teamestimatedmetrics.TeamEstimatedMetrics(season=year_string).get_data_frames()[0]
        t_df['SEASON_YEAR'] = year_string
        df = pd.concat([df, t_df])
    
    return df

def create_avg_over_season_columns(source: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """
    @parameter source: A dataframe containing atleast 1 season of data from the playergamelogs endpoint in the api
    @parameter names: a list of columns from the dataframe you wish to average
    
    Will create a new dataframe with player_id, game_id (unique identifiers) and the averages of the stats you request
    """
    # Sort the DataFrame (optional depending on your needs)
    source = source.sort_values(by=['PLAYER_ID', 'SEASON_YEAR'])

    new_column_names = []
    for name in names:
        season_avg = source.groupby(['PLAYER_ID', 'SEASON_YEAR'])[name].mean()
        column_name = f"{name}_SEASON_AVG"
        new_column_names.append(column_name)
    
        # Map the calculated averages back to the original DataFrame
        source[column_name] = source.set_index(['PLAYER_ID', 'SEASON_YEAR']).index.map(season_avg)

    new_column_names.extend(['PLAYER_ID', 'GAME_ID'])
    # Return the desired DataFrame with PLAYER_ID, GAME_ID, and the new column
    return source[new_column_names]

def weighted_average(series):
    weights = np.arange(10, 0, -1)
    if len(series) < len(weights):
        current_weights = weights[:len(series)]
    else:
        current_weights = weights
    return np.average(series, weights=current_weights)

def create_weighted_average_columns(source: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    """
    @parameter source: A dataframe from the playergamelogs endpoint of the nba_api
    @parameter names: The names of the columns we are calculating WMA for
    """
    source = source.sort_values(by=['PLAYER_ID', 'SEASON_YEAR', 'GAME_DATE'])

    wma_names = []
    for name in names:
        wma_col_name = f"WMA_{name}_LAST_3"
        wma_names.append(wma_col_name)

        source[wma_col_name] = (
            source.groupby(['PLAYER_ID', 'SEASON_YEAR'])[name]
            .shift(1)
            .rolling(window=3, min_periods=1)
            .apply(weighted_average, raw=True)
        )
    
    wma_names.extend(['PLAYER_ID', 'GAME_ID'])

    return source[wma_names]

def create_useage_rate_column(p_source: pd.DataFrame, t_source: pd.DataFrame) -> pd.DataFrame:

    t = t_source.rename(columns={'FTA': 'TFTA', 'FGA': 'TFGA', 'TOV': 'TTOV'})
    t['TMIN'] = t['MIN'] * 5

    useagelogs = p_source.merge(right=t[['TEAM_ID', 'GAME_ID', 'TFGA', 'TFTA', 'TTOV', 'TMIN']], how='left', on=['GAME_ID', 'TEAM_ID'])

    useagelogs['USEAGE_RATE'] = ((100 * (useagelogs['FGA']) + (0.44 * useagelogs['FTA']) 
                              + useagelogs['TOV']) * useagelogs['TMIN']) / ((useagelogs['TFGA'] + (0.44 * useagelogs['TFTA']) + useagelogs['TTOV']) * 5 * useagelogs['MIN'])
    
    useagelogs['UR_LAST_5'] = (
        useagelogs.groupby(['PLAYER_ID', 'SEASON_YEAR'])['USEAGE_RATE']
        .rolling(window=5, min_periods=1)
        .apply(np.average, raw=True)
        .reset_index(level=[0, 1], drop=True)
    )

    return useagelogs[['PLAYER_ID', 'GAME_ID', 'UR_LAST_5']]


def create_fatigue_columns(source: pd.DataFrame) -> pd.DataFrame:
    """
    @parameter source: A dataframe from the playergamelogs nba_api endpoint

    Will give you three new columns 'HOME' a 1/0 boolean if the
    game is at home or not.

    'GAMES_LAST_7_DAYS' an int with how many games in the past 7 days
    'AWAY_GAMES_IN_A_ROW' used to determine amount of time on the row 
    """
    source['GAME_DATE'] = pd.to_datetime(source['GAME_DATE'])
    source['HOME'] = np.where(source['MATCHUP'].str.contains('@'), 0, 1)

    def calc_games_last_7(group):
        counts = []
        for i in range(len(group)):
            current_date = group.iloc[i]['GAME_DATE']
            # Filter rows strictly before the current date and within the past 7 days
            past_week_games = group[(group['GAME_DATE'] < current_date) & 
                                (group['GAME_DATE'] >= current_date - pd.Timedelta(days=7))]
            counts.append(len(past_week_games))
        return pd.Series(counts, index=group.index)

    source = source.sort_values(by=['PLAYER_ID', 'GAME_DATE'])
    source['GAMES_LAST_7_DAYS'] = (
        source
        .groupby('PLAYER_ID', group_keys=False)
        .apply(calc_games_last_7)
    )

    def calc_away_streak(group):
        # Initialize the result list to store away game streaks
        away_streak = []
        current_streak = 0  # To keep track of the ongoing streak

        for i in range(len(group)):
            if group.iloc[i]['HOME'] == 0:
                # Increment the streak if HOME = 0
                current_streak += 1
            else:
                # Reset the streak if HOME = 1
                current_streak = 0
            # Append the current streak to the result
            away_streak.append(current_streak)

        return pd.Series(away_streak, index=group.index)
    
    source['AWAY_GAMES_IN_A_ROW'] = source.groupby('PLAYER_ID', group_keys=False).apply(calc_away_streak)

    source.sort_values(by=['PLAYER_ID', 'GAME_DATE'], inplace=True)

    source['MIN_LAST_3_DAYS'] = (
        source.groupby('PLAYER_ID', group_keys=False)  # Prevents extra index levels
        .apply(lambda x: 
            x.groupby('GAME_DATE', as_index=False)['MIN'].sum()  # Aggregate by GAME_DATE
            .set_index('GAME_DATE')['MIN']  # Now, each date has a single value
            .rolling('3D', closed='left')  # 3-day rolling sum excluding current game
            .sum()
        ).fillna(0)
    ).reset_index(level=0, drop=True) 

    return source[['PLAYER_ID', 'GAME_ID', 'HOME', 'GAMES_LAST_7_DAYS','AWAY_GAMES_IN_A_ROW', 'MIN_LAST_3_DAYS']]

def create_hist_perf_columns(source: pd.DataFrame) -> pd.DataFrame:
    source.rename(columns={'TEAM_ABBREVIATION': 'P_TEAM_ABBR'}, inplace=True)
    source['A_TEAM_ABBR'] = source['MATCHUP'].str.split(' ').str[-1]
    teams_df = teams.get_teams()
    teams_df = pd.DataFrame(teams_df)
    source = source.merge(teams_df[['id', 'abbreviation']], left_on='A_TEAM_ABBR', right_on='abbreviation', how='left')
    source.rename(columns={'id': 'A_TEAM_ID'}, inplace=True)
    source.drop(columns=['abbreviation'], inplace=True)

    source['GAME_DATE'] = pd.to_datetime(source['GAME_DATE'])
    source = source.sort_values(by=['PLAYER_ID', 'A_TEAM_ID', 'GAME_DATE'])

    historic_vs_team = []
    for index, row in source.iterrows():

        past_games = source[
            (source['PLAYER_ID'] == row['PLAYER_ID']) &
             (source['A_TEAM_ID'] == row['A_TEAM_ID']) &
             (source['GAME_DATE'] < row['GAME_DATE'])
        ]

        if past_games.empty:
            # if there are no actual past games, look into any time performances
            games = source[
            (source['PLAYER_ID'] == row['PLAYER_ID']) &
             (source['A_TEAM_ID'] == row['TEAM_ID'])
            ]
            avg = games['PTS'].mean()
        else:
            avg = past_games['PTS'].mean()
        historic_vs_team.append(avg)

    source['HISTORIC_VS_TEAM'] = historic_vs_team
    return source[['PLAYER_ID', 'GAME_ID', 'HISTORIC_VS_TEAM']]

def calc_team_stats(t_source: pd.DataFrame) -> pd.DataFrame:
    t_source = t_source.sort_values(by=['TEAM_ID', 'SEASON_YEAR', 'GAME_DATE']).reset_index(drop=True)
    t_source['WIN_INDICATOR'] = (t_source['WL'] == 'W').astype(int)
    
    t_source['WINS_PER_LAST_10'] = (
        t_source.groupby(['TEAM_ID', 'SEASON_YEAR'])['WIN_INDICATOR']
        .rolling(window=10, min_periods=1)  # Rolling window of size 10
        .apply(lambda x: x.mean(), raw=False)
        .shift(1)  # Exclude the current game from the count
        .reset_index(drop=True)  # Align indices with the original DataFrame
    )

    t_source['OPP_TEAM_ABBR'] = t_source['MATCHUP'].apply(lambda x: x.split(' ')[-1])

    # Step 2: Create an opponent DataFrame with GAME_ID and PTS (linking via OPP_TEAM_ABBR)
    opp_df = t_source[['GAME_ID', 'TEAM_ABBREVIATION', 'PTS']].rename(columns={
        'TEAM_ABBREVIATION': 'OPP_TEAM_ABBR',  # Use 'TEAM_ABBR' to align with the extracted 'OPP_TEAM_ABBR'
        'PTS': 'PTS_ALLOWED'          # Rename PTS to PTS_ALLOWED
    })

    t_source = t_source.merge(opp_df, how='left', on=['GAME_ID', 'OPP_TEAM_ABBR'])
    
    t_source['WMA_PTS_ALLOWED_10'] = (
        t_source.groupby(['TEAM_ID', 'SEASON_YEAR'])['PTS_ALLOWED']
        .rolling(window=10, min_periods=1)
        .apply(weighted_average, raw=True)
        .reset_index(level=[0, 1], drop=True)
    )

    t_source['WMA_AST_10'] = (
        t_source.groupby(['TEAM_ID', 'SEASON_YEAR'])['AST']
        .rolling(window=10, min_periods=1)
        .apply(weighted_average, raw=True)
        .reset_index(level=[0, 1], drop=True)
    )

    season_avg = t_source.groupby(['TEAM_ID', 'SEASON_YEAR'])['PTS_ALLOWED'].mean()
    t_source['PTS_ALLOWED_OVR_SEASON'] = t_source.set_index(['TEAM_ID', 'SEASON_YEAR']).index.map(season_avg)

    return t_source[['TEAM_ID', 'GAME_ID', 'WINS_PER_LAST_10', 'WMA_PTS_ALLOWED_10', 'WMA_AST_10', 'PTS_ALLOWED_OVR_SEASON']]

def create_team_stats_columns(p_source: pd.DataFrame, t_source, est_df: pd.DataFrame) -> pd.DataFrame:
    t_source = calc_team_stats(t_source)

    p_source = p_source.merge(t_source[['TEAM_ID', 'GAME_ID', 'WMA_AST_10']], how='left', on=['GAME_ID', 'TEAM_ID'])

    p_source['OPP_TEAM_ABBR'] = p_source['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    
    team_df = pd.DataFrame(teams.teams)
    p_source = p_source.merge(team_df[[0, 1]], how='left', left_on='OPP_TEAM_ABBR', right_on=1)

    p_source.rename(columns={0: 'OPP_TEAM_ID'}, inplace=True)
    p_source.drop(columns=['OPP_TEAM_ABBR', 1], inplace=True)

    p_source = p_source.merge(t_source[['GAME_ID', 'TEAM_ID', 
                                        'WINS_PER_LAST_10', 'WMA_PTS_ALLOWED_10', 
                                        'PTS_ALLOWED_OVR_SEASON']], how='left', left_on=['GAME_ID', 'OPP_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'])

    p_source.rename(columns={
        'WINS_PER_LAST_10': 'OPP_WINS_LAST_10',
        'WMA_PTS_ALLOWED_10': 'OPP_WMA_PTS_ALLOWED',
        'PTS_ALLOWED_OVR_SEASON': 'OPP_PTS_ALLOWED'
    }, inplace=True)

    # Include player teams pace and opp teams pace

    p_source.drop(columns=['TEAM_ID_y'], inplace=True)
    p_source.rename(columns={
        'TEAM_ID_x': 'TEAM_ID'
    }, inplace=True)

    p_source = p_source.merge(est_df[['TEAM_ID', 'E_PACE', 'E_OFF_RATING', 'SEASON_YEAR']], how='left', on=['TEAM_ID', 'SEASON_YEAR'])
    p_source.rename(columns={'E_PACE': 'PLAYER_TEAM_PACE',
                           'E_OFF_RATING': 'PLAYER_TEAM_OFF_RATING'}, inplace=True)
    
    p_source = p_source.merge(est_df[['TEAM_ID', 'E_PACE', 'E_DEF_RATING', 'SEASON_YEAR']], how='left', 
                              right_on=['TEAM_ID', 'SEASON_YEAR'], left_on=['OPP_TEAM_ID', 'SEASON_YEAR'])
    p_source.rename(columns={
        'E_PACE': 'OPP_TEAM_PACE',
        'E_DEF_RATING': 'OPP_DEF_RATING'
    }, inplace=True)
    
    return p_source[['GAME_ID', 'PLAYER_ID', 'WMA_AST_10', 'OPP_WINS_LAST_10', 
                     'OPP_WMA_PTS_ALLOWED', 'OPP_PTS_ALLOWED', 'PLAYER_TEAM_PACE', 
                     'PLAYER_TEAM_OFF_RATING', 'OPP_TEAM_PACE', 'OPP_DEF_RATING']]



def create_training_data() -> pd.DataFrame:

    if os.path.exists('playergamelogs.csv'):
        pgl_df = pd.read_csv('playergamelogs.csv')
    else:
        pgl_df = create_playergamelogs_df()
        pgl_df.to_csv('playergamelogs.csv')

    if os.path.exists('teamgamelogs.csv'):
        tgl_df = pd.read_csv('teamgamelogs.csv')
    else:
        tgl_df = create_teamgamelogs_df()
        tgl_df.to_csv('teamgamelogs.csv')
    
    if os.path.exists('estimatedteammetrics.csv'):
        etm_df = pd.read_csv('estimatedteammetrics.csv')
    else:
        etm_df = create_estimatedteammetrics_df()
        etm_df.to_csv('estimatedteammetrics.csv')

    
    train_df = pgl_df[['PLAYER_ID', 'GAME_ID', 'PTS']]

    cols_to_avg = ['PTS', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'STL', 'BLK']
    avg_df = create_avg_over_season_columns(pgl_df, cols_to_avg)

    wma_names = ['PTS', 'FGA', 'FG3A', 'FTA', 'MIN', 'FT_PCT']
    wma_df = create_weighted_average_columns(pgl_df, wma_names)

    fat_df = create_fatigue_columns(source=pgl_df)
    hist_df = create_hist_perf_columns(source=pgl_df)
    team_stats_df = create_team_stats_columns(pgl_df, tgl_df, etm_df)
    ur_df = create_useage_rate_column(pgl_df, tgl_df)

    train_df = train_df.merge(avg_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(wma_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(fat_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(hist_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(team_stats_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(ur_df, how='left', on=['PLAYER_ID', 'GAME_ID'])

    train_df['EST_GAME_PACE'] = (train_df['PLAYER_TEAM_PACE'] + train_df['OPP_TEAM_PACE']) / 2
    train_df.drop(columns=['PLAYER_TEAM_PACE', 'OPP_TEAM_PACE'], inplace=True)

    return train_df

def save_training_data():
    train_df = create_training_data()

    train_df.to_csv('train.csv')

if __name__ == '__main__':
    save_training_data()