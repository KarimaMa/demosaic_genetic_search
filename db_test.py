from peewee import *

if __name__ == "__main__":
    db_host = 'mysql.csail.mit.edu'
    db_name = 'ModelSearch'
    db_user = 'karima'
    db_password = 'trisan4th' #password
    db_charset = 'utf8mb4'

    db_conn = {
        'host': db_host,
        'user': db_user,
        'passwd': db_password,
        'port': 3306,
    }

    db = MySQLDatabase(db_name, **db_conn)

    db.connect()

    # print(db)
    # import ipdb; ipdb.set_trace()

