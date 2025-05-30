import pandas as pd
import numpy as np
from typing import List
import re


#####################################
####------ Base function -------#####
#####################################
def create_season_stat_columns(source: pd.DataFrame, names: List[str], prefix: str, stat_func):
    """
    Used for filling NaN values in rolling window based columns
    """
    source = source.sort_values(by=['PLAYER_ID', 'SEASON_YEAR'])

    new_column_names = []
    for name in names:
        season_avg = source.groupby(['PLAYER_ID', 'SEASON_YEAR'])[name].apply(stat_func)
        column_name = f"SEASON_{prefix}_{name}"
        new_column_names.append(column_name)
    
        # Map the calculated averages back to the original DataFrame
        source[column_name] = source.set_index(['PLAYER_ID', 'SEASON_YEAR']).index.map(season_avg)

    new_column_names.extend(['PLAYER_ID', 'GAME_ID'])
    # Return the desired DataFrame with PLAYER_ID, GAME_ID, and the new column
    return source[new_column_names]



def create_stat_columns(source: pd.DataFrame, names: List[str], win_length: int, prefix: str, stat_func):
    """
    @parameter source: A dataframe from the playergamelogs endpoint of the nba_api
    @parameter names: The names of the columns we are applying the function to
    @parameter win_length: The length of the window 'last 3, 5, 10' 
    @parameter prefix: Used for namming convetion of the columns
    @parameter stat_func: np.std, np.mean, weighted_average, exp_average (etc.)
    """
    source = source.sort_values(by=['PLAYER_ID', 'SEASON_YEAR', 'GAME_DATE'])

    new_names = []
    for name in names:
        new_col_name = f"{prefix}_{name}_{win_length}"
        new_names.append(new_col_name)

        source[new_col_name] = (
            source.groupby(['PLAYER_ID', 'SEASON_YEAR'], group_keys=False)
            .apply(lambda g: g[name].shift(1)
                         .rolling(window=win_length, min_periods=1)
                         .apply(stat_func, raw=True))            
        )

    new_names.extend(['PLAYER_ID', 'GAME_ID'])

    if stat_func == np.std:
        temp_df = create_season_stat_columns(source, names, prefix, stat_func)
    else:
        temp_df = create_season_stat_columns(source, names, prefix, np.mean)
    
    source = source.merge(temp_df, on=['PLAYER_ID', 'GAME_ID'], how='left')

    s = -2 if win_length < 10 else -3

    for col in source.columns:
        if col.startswith(prefix):
            attr_name = col[4:s]
            season_stat_col = f'SEASON_{prefix}_{attr_name}'
            if season_stat_col in source.columns:
                source[col].fillna(source[season_stat_col], inplace=True)


    source.drop(columns=[col for col in source.columns if col.startswith(f'SEASON_{prefix}')], inplace=True)

    return source[new_names]


#######################################################
#---------------- WEIGHTED AVERAGE FUNCTIONS ---------#
#######################################################
def get_linear_weights(window):
    weights = np.arange(window, 0, -1)
    return weights / weights.sum()

def weighted_average(series):
    series = series[~np.isnan(series)]  # remove NaNs

    if len(series) == 0:
        return np.nan

    weights = get_linear_weights(len(series))

    # trim weights if not enough values
    return np.dot(series, weights) / sum(weights)

########################################################
#----------- Fatigue/pscycological features -----------#
########################################################

# Home flag
# Adjusted home/away performance
# playoff flag
def create_pysch_features(source: pd.DataFrame):
    # HOME field
    source['HOME'] = np.where(source['MATCHUP'].str.contains('@'), 0, 1)

    season_avg_pts = (
        source.groupby(['PLAYER_ID', 'SEASON_YEAR'])['PTS']
        .mean()
        .reset_index()
        .rename(columns={'PTS': 'SEASON_AVG_PTS'})
    )

    home_away_avg = (
        source.groupby(['PLAYER_ID', 'SEASON_YEAR', 'HOME'])['PTS']
        .mean()
        .unstack(fill_value=np.nan)  # HOME=0 (away), HOME=1 (home)
        .rename(columns={0: 'AWAY_PERF', 1: 'HOME_PERF'})
        .reset_index()
    )

    source = source.merge(home_away_avg, on=['PLAYER_ID', 'SEASON_YEAR'], how='left')
    source = source.merge(season_avg_pts, on=['PLAYER_ID', 'SEASON_YEAR'], how='left')


    # calculates how different the player is playing at home vs away
    source['HOME_AWAY_DIFF'] = source['HOME_PERF'] - source['AWAY_PERF']
    source['HOME_AWAY_DIFF'].fillna(0, inplace=True)

    source['HOME_AWAY_RATIO'] = np.where(
        source['SEASON_AVG_PTS'] > 0,
        source['HOME_AWAY_DIFF'] / source['SEASON_AVG_PTS'],
        0
    )

    q3 = source['HOME_AWAY_RATIO'].abs().quantile(0.75)

    source['HOME_AWAY_MEANINGFUL'] = (source['HOME_AWAY_RATIO'].abs() > q3).astype(int)
    source['HOME_AWAY_RATIO'] = source['HOME_AWAY_RATIO'].clip(-1.0, 1.0)

    return source[['HOME_AWAY_RATIO', 'HOME_AWAY_MEANINGFUL','HOME', 'PLAYER_ID', 'GAME_ID']]




def create_useage_rate_column(p_source: pd.DataFrame, t_source: pd.DataFrame) -> pd.DataFrame:

    t = t_source.rename(columns={'FTA': 'TFTA', 'FGA': 'TFGA', 'TOV': 'TTOV'})
    t['TMIN'] = t['MIN'] * 5

    useagelogs = p_source.merge(right=t[['TEAM_ID', 'GAME_ID', 'TFGA', 'TFTA', 'TTOV', 'TMIN']], how='left', on=['GAME_ID', 'TEAM_ID'])

    useagelogs['UR'] = ((100 * (useagelogs['FGA']) + (0.44 * useagelogs['FTA']) 
                              + useagelogs['TOV']) * useagelogs['TMIN']) / ((useagelogs['TFGA'] + (0.44 * useagelogs['TFTA']) + useagelogs['TTOV']) * 5 * useagelogs['MIN'])
    
    useagelogs.fillna(0, inplace=True)
    return useagelogs[['UR', 'GAME_ID', 'PLAYER_ID']]

def main():
    team_df = pd.read_csv('nba_api_data/teamgamelogs.csv')
    player_df = pd.read_csv('nba_api_data/playergamelogs.csv')

    player_df = player_df.merge(right=create_useage_rate_column(player_df, team_df), how='left', on=['PLAYER_ID', 'GAME_ID'])

    train_df = player_df[['PLAYER_ID', 'GAME_ID', 'PTS']]

    player_df['FG2A'] = player_df['FGA'] - player_df['FG3A']

    col_names = ['PTS', 'FG2A', 'FG3A', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'UR']

    wma_3_df = create_stat_columns(player_df, col_names, 3, 'WMA', weighted_average)
    wma_5_df = create_stat_columns(player_df, col_names, 5, 'WMA', weighted_average)

    col_names.append('OREB')
    sma_10_df = create_stat_columns(player_df, col_names, 10, 'AVG', np.mean)

    col_names = ['PTS', 'MIN', 'FG_PCT', 'FG3_PCT', 'UR']
    std_10_df = create_stat_columns(player_df, col_names, 10, 'STD', np.std)

    psy_df = create_pysch_features(player_df)

    starter_df = pd.read_csv('nba_api_data/lineupdata.csv')
    starter_df = starter_df[['GAME_ID', 'PLAYER_ID', 'STARTED']]
    train_df = train_df.merge(starter_df, how='left', on=['PLAYER_ID', 'GAME_ID'])


    train_df = train_df.merge(wma_3_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(wma_5_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(sma_10_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(std_10_df, how='left', on=['PLAYER_ID', 'GAME_ID'])
    train_df = train_df.merge(psy_df, how='left', on=['PLAYER_ID', 'GAME_ID'])

    # derivative feautures, will be made into seperate function at some point
    train_df['RATIO_2_to_3'] = train_df['AVG_FG2A_10'] / train_df['AVG_FG3A_10']
    train_df['VOLATILITY_RATIO_PTS'] = train_df['STD_PTS_10'] / train_df['AVG_PTS_10']
    train_df['VOLATILITY_RATIO_PTS'].fillna(0, inplace=True)
    train_df['UNPREDICTABLE'] = (train_df['VOLATILITY_RATIO_PTS'] > 0.4).astype(int)
    train_df['HEATING_OR_COOLING'] = train_df['WMA_PTS_3'] - train_df['AVG_PTS_10']
    train_df['CHANGE_OF_ROLE'] = train_df['WMA_MIN_3'] - train_df['AVG_MIN_10']
    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_df.fillna(0, inplace=True)

    train_df.to_csv('train.csv')

if __name__ == "__main__":
    main()