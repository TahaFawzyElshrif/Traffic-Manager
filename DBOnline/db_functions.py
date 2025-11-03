import sqlite3
import random
import json
# Suppose chosen model is highgroubed (just affect action space here to be determainted),stefano,literature,d3qn (any one),mapping_duration should be known here
#  action space is from 0-8 (first  red for different duration ,second green for  the same diferent duration)


def create_database(db_path,TABLE_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
    DROP TABLE IF  EXISTS {TABLE_NAME};
    """)
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        state_before TEXT,
        action INTEGER,
        state_after TEXT
    );
    """)

    conn.commit()
    conn.close()


def generate_random_state():
    """
    Generate a random traffic state that looks realistic.
    Features:
    - avg_speed (km/h): typically 0–60
    - var_speed: 0–10
    - avg_waiting_time (sec): 0–300
    - var_waiting_time: 0–100
    - avg_throughput (veh/min): 0–50
    - avg_queue_length (veh): 0–50
    - avg_occupancy (%): 0–100
    """
    return (
       round(random.uniform(5, 60), 2),
       round(random.uniform(0, 10), 2),
       round(random.uniform(0, 300), 2),
       round(random.uniform(0, 100), 2),
       round(random.uniform(0, 50), 2),
       round(random.uniform(0, 50), 2),
       round(random.uniform(0, 100), 2)
    )

def insert_sample_data(TABLE_NAME,db_path, samples,n_action):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for _ in range(samples):
        state_before = generate_random_state()
        state_after = generate_random_state()
        action = random.randint(0, n_action)

        cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (state_before, action, state_after)
        VALUES (?, ?, ?);
        """, (json.dumps(state_before), action, json.dumps(state_after)))

    conn.commit()
    conn.close()

def load_data(TABLE_NAME,db_path): #load all data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {TABLE_NAME};")
    data = cursor.fetchall()
    conn.close()
    return data


# Run these once

