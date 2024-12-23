import sqlite3
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelogs


def create_players_table():
    connection = sqlite3.connect("moneylinedb.db")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            is_active BOOLEAN
        )
    """)

    connection.commit()

    all_players = players.get_players()
    formatted = []
    for p in all_players:
        formatted.append((p["id"], p["first_name"], p["last_name"], p["is_active"]))

    cursor.executemany('''
        INSERT INTO players (id, first_name, last_name, is_active)
        VALUES (?, ?, ?, ?)
    ''', formatted)

    connection.commit()

def create_gamelogs_table():
    connection = sqlite3.connect("moneylinedb.db")
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gamelogs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        player_id INT NOT NULL,
        team_id INT NOT NULL,
        game_id INT NOT NULL,
        game_date DATE NOT NULL,
        min FLOAT(13) NOT NULL,
        fg_pct FLOAT(13),
        fg3_pct FLOAT(13),
        ft_pct FLOAT(13),
        reb INT,
        ast INT,
        tov INT,
        stl INT,
        blk INT,
        pf INT,
        pts INT,
        plus_minus INT,
        available_flag INT
        )
    """)
    
    connection.commit()

    start_year = 17
    end_year = 18
    formatted = []
    while start_year < 25:
        year_filter = f"20{start_year}-{end_year}"
        endpoint = playergamelogs.PlayerGameLogs(season_nullable=year_filter)
        rows = endpoint.get_dict()["resultSets"][0]["rowSet"]

        for row in rows:
            formatted.append((row[1], row[4], row[7], row[8], row[11], row[14], row[17], row[20], 
                              row[23], row[24], row[25], row[26], row[27], row[29], row[31], row[32], row[-2]))

        start_year += 1
        end_year += 1
    
    cursor.executemany("""
    INSERT INTO gamelogs (player_id, team_id, game_id, game_date, min, fg_pct, fg3_pct, ft_pct, reb, ast, tov, stl, blk, pf, pts, plus_minus, available_flag)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, formatted)

    connection.commit()
    
    

def main():
    create_gamelogs_table()
    print("hi")
    

if __name__ == "__main__":
    main()
