import sqlite3

conn = sqlite3.connect('student.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', tables)

if tables:
    for table in tables:
        print(f"\nTable: {table[0]}")
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        print("Columns:", columns)
        cursor.execute(f"SELECT * FROM {table[0]}")
        rows = cursor.fetchall()
        print("Rows:", rows)

conn.close()
