import sqlite3



def create_table():
    conn = sqlite3.connect('cyclethief.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cycle_owner_theif (
        Id INTEGER PRIMARY KEY AUTOINCREMENT,
        Cycle_ID INTEGER,
        Owner_ID INTEGER,
        Thief_ID INTEGER,
        Status INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(data):
    conn = sqlite3.connect('cyclethief.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO cycle_owner_theif (Cycle_ID, Owner_ID, Thief_ID, Status)
        VALUES (?, ?, ?,?)
    ''', data)
    conn.commit()
    conn.close()

def read_data(status):
    conn = sqlite3.connect('cyclethief.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM cycle_owner_theif
        WHERE Status = ?
    ''', (status,))
    rows = cursor.fetchall()
    conn.commit()
    conn.close()
    return rows


def update_data(id_list,new_status):
    conn = sqlite3.connect('cyclethief.db')
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE cycle_owner_theif
    SET Status = ?
    WHERE Id IN ({})
    '''.format(','.join('?' for _ in id_list)), [new_status] + id_list)
    conn.commit()
    conn.close()

# create_table()
# insert_data((1,10,10,1))
# insert_data((2,13,23,1))
# insert_data((1,10,20,1))
# insert_data((0,10,20,0))
# insert_data((0,10,20,0))
# insert_data((0,10,20,0))
# update_data(id_list=[1,7,8], new_status=1)
# print(read_data(status=1))


