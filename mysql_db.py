from peewee import *
from tree import *
from model_lib import *
import datetime



def create_table(password):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db


  class SeenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)

  db.create_tables([SeenTrees])

  tables = db.get_tables()
  print(tables)

  basic1d = basic1D_green_model()
  basic2d = basic2D_green_model()
  multires1d = multires_green_model()

  key = 0
  tree_hash = str(hash(basic1d)).zfill(30)
  basic1d_record = SeenTrees.create(model_id=key, tree_hash=tree_hash, machine="tefnut", \
                                  experiment_dir="seed", tree_id_str=basic1d.id_string())
  basic1d_record.save()
  query = SeenTrees.select()
  for t in query:
    print(t.model_id, t.add_date, t.tree_hash, t.tree_id_str, t.machine, t.experiment_dir)


def select(password):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class SeenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)

  query = SeenTrees.select()
  print(len(query))
  for t in query:
    print(t.model_id, t.add_date, t.tree_hash, t.tree_id_str, t.machine, t.experiment_dir)


def find(password, tree_hash, tree_id_string, logger):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class SeenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)

  query = SeenTrees.select().where(SeenTrees.tree_hash == tree_hash, \
                                   SeenTrees.tree_id_str == tree_id_string)
  if len(query) != 0:
    logger.info("---------")
    if len(query) > 1:
      logger.info(f"ERROR: tree occurs in SeenTrees more than once!")
      
    logger.info("Already seen tree:")
    for t in query:
      logger.info(f"ModelId: {t.model_id} Machine: {t.machine} Dir: {t.experiment_dir} \
                    hash: {t.tree_hash} id_str: {t.tree_id_str} date: {t.add_date}")
    logger.info("---------")
    return True
  return False


def mysql_insert(password, model_id, machine, exp_dir, tree_hash, id_str):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class SeenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)

  tree_hash = str(tree_hash).zfill(30)
  record = SeenTrees.create(model_id=model_id, tree_hash=tree_hash, machine=machine, \
                                  experiment_dir=exp_dir, tree_id_str=id_str)
  record.save()


if __name__ == "__main__":
  # print("inserting seed tree")
  # create_table("trisan4th")
  import util 
  import logging
  log_format = '%(asctime)s %(levelname)s %(message)s'
  logger = util.create_logger(f'mysql_logger', logging.INFO, log_format, f'mysql_log')
  print("checking insertion worked...")
  select("trisan4th")
  idstr = "GreenExtractor-1-1,1-SumR-1-16-Mul-16-16,16-Conv1D-16-1-Input-1-1---Softmax-16-16-Conv1x1-16-16-Relu-16-16-Conv1x1-16-16-Relu-16-16-Conv1D-16-1-Input-1-1----------Input-1-1--"
  find("trisan4th", "000000000000338642508656442816", idstr, logger)
