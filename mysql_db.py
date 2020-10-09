from peewee import *
from tree import *
from model_lib import *
import datetime



def drop_tables(password):
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

  db.drop_tables((SeenTrees,))
  print("tables in drop")
  tables = db.get_tables()
  print(tables)


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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  db.create_tables([SeenTrees])

  tables = db.get_tables()
  print(tables)

  basic1d = basic1D_green_model()
  basic2d = basic2D_green_model()
  multires1d = multires_green_model()

  key = 0
  tree_hash = str(hash(basic1d)).zfill(30)
  basic1d_record = SeenTrees.create(model_id=key, tree_hash=tree_hash, machine="tefnut", \
                                  experiment_dir="seed", tree_id_str=basic1d.id_string(), psnr_0=31.38, psnr_1=-1, psnr_2=-1)
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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  query = SeenTrees.select()
  print(len(query))
  for t in query:
    print(t.model_id, t.tree_hash, t.machine, t.experiment_dir, t.psnr_0, t.psnr_1, t.psnr_2)


def select_range(password, id_min, id_max):
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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  query = SeenTrees.select().where(SeenTrees.model_id < id_max, SeenTrees.model_id > id_min)
  print(len(query))
  for t in query:
    print(t.model_id, t.tree_hash, t.machine, t.experiment_dir, t.psnr_0, t.psnr_1, t.psnr_2)



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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  tree_hash = str(tree_hash).zfill(30)
  query = SeenTrees.select().where(SeenTrees.tree_hash == tree_hash, \
                                   SeenTrees.tree_id_str == tree_id_string)
  if len(query) != 0:
    logger.info("---------")
    if len(query) > 1:
      logger.info(f"ERROR: tree occurs in SeenTrees more than once!")
      
    logger.info("Already seen tree:")
    for t in query:
      psnrs = [t.psnr_0, t.psnr_1, t.psnr_2]
      logger.info(f"ModelId: {t.model_id} Machine: {t.machine} Dir: {t.experiment_dir} " +
                  f"hash: {t.tree_hash} id_str: {t.tree_id_str} date: {t.add_date}")
      logger.info("---------")
      return psnrs
  logger.info(f"hash {tree_hash} not in database")
  return None

def mysql_insert(password, model_id, machine, exp_dir, tree_hash, id_str, psnrs, logger):
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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  tree_hash = str(tree_hash).zfill(30)

  already_in_db = find(password, tree_hash, id_str, logger)
  if already_in_db is None:
    record = SeenTrees.create(model_id=model_id, tree_hash=tree_hash, machine=machine, \
                                    experiment_dir=exp_dir, tree_id_str=id_str, \
                                    psnr_0=psnrs[0], psnr_1=psnrs[1], psnr_2=psnrs[2])
    record.save()

  else:
    logger.info(f"other machine also generated tree with model {model_id}'s hash {tree_hash}")


def mysql_delete(password):
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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  found = SeenTrees.select().where(SeenTrees.model_id != 0)
  for f in found:
    f.delete_instance()

def mysql_delete(password, id_min, id_max):
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
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  found = SeenTrees.select().where(SeenTrees.model_id >= id_min, SeenTrees.model_id <= id_max)
  for t in found:
    print(t.model_id, t.tree_hash, t.machine, t.experiment_dir, t.psnr_0, t.psnr_1, t.psnr_2)
    t.delete_instance()


if __name__ == "__main__":
  # print("inserting seed tree")
  # drop_tables("trisan4th")
  # create_table("trisan4th")

  import util 
  import logging
  log_format = '%(asctime)s %(levelname)s %(message)s'
  logger = util.create_logger(f'mysql_logger', logging.INFO, log_format, f'mysql_log')
  #mysql_delete("trisan4th")
  #mysql_delete("trisan4th", 601, 640)
  #select_range("trisan4th", 600, 1000)
  print("checking insertion worked...")
  select("trisan4th")
  #idstr = "GreenExtractor-1-1,1-SumR-1-16-Mul-16-16,16-Conv1D-16-1-Input-1-1---Softmax-16-16-Conv1x1-16-16-Relu-16-16-Conv1x1-16-16-Relu-16-16-Conv1D-16-1-Input-1-1----------Input-1-1--"
  #find("trisan4th", "000000000000338642508656442816", idstr, logger)
