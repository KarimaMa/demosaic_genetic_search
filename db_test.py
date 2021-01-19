import peewee as pw

if __name__ == "__main__":
    db_conn = {
        'host': 'modelsearchinstance.c3t2omr0tk0e.us-east-2.rds.amazonaws.com',
        'user': 'admin',
        'passwd': 'trisan4th',
        'port': 3306,
    }

    db = pw.MySQLDatabase('modelsearch', **db_conn)
    print("Created DB")

    class MySQLModel(pw.Model):
        """A base model that will use our MySQL database"""
        class Meta:
            database = db

    class User(MySQLModel):
        username = pw.CharField()

    print("connecting")
    conn = db.connect()
    db.close()
    print("done")
