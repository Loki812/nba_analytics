import sqlite3
from nba_api.stats.static import players


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

def main():
    connection = sqlite3.connect("moneylinedb.db")

    cursor = connection.cursor()

    all_players = players.get_players()
    formatted = []
    for p in all_players:
        formatted.append((p["id"], p["first_name"], p["last_name"], p["is_active"]))

    cursor.executemany('''
        INSERT INTO players (id, first_name, last_name, is_active)
        VALUES (?, ?, ?, ?)
    ''', formatted)

    connection.commit()
    

if __name__ == "__main__":
    main()