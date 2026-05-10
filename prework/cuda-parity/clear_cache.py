import sqlite3, os
db = os.path.expanduser('~/.cache/tinygrad/cache.db')
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('tables:', tables)
for t in tables:
    if 'abduct' in t.lower() or 'beam' in t.lower():
        cur.execute(f"DROP TABLE '{t}'")
        print(f'dropped {t}')
conn.commit()
conn.close()
