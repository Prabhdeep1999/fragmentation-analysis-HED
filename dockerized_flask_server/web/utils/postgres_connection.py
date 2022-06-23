import psycopg2

# local import
from .config import config

def connect():
    """Connect to the PostgreSQL database server

    Returns:
        cur (_cursor): Returns Cursor object of Postgresql database
        conn (connection): Return Connetion object of Postgresql database
    """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
        return cur, conn
    except Exception as e:
        print(e, 1)

# Connection with database
cursor, connection = connect()