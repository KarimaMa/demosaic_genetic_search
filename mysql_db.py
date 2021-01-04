from peewee import *
from tree import *
from model_lib import *
import datetime
import argparse
import demosaic_ast 

def drop_table(password, tablename):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  if tablename == "testgreen":
    table_to_drop = TestGreenTrees
  elif tablename == "green":
    table_to_drop = GreenTrees
  else:
    table_to_drop = ChromaTrees

  db.drop_tables((table_to_drop,))
  tables = db.get_tables()
  print("tables remaining after drop")
  print(tables)


def create_table(password, tablename):

  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()


  if tablename == "testgreen":
    table = TestGreenTrees
    models = [MultiresQuadGreenModel(2,10), GreenDemosaicknet(3,8)]
    psnrs = [34.10, 34.19]
  elif tablename == "green":
    table = GreenTrees
    models = [MultiresQuadGreenModel(2,10), GreenDemosaicknet(3,8)]
    psnrs = [34.10, 34.19]
  else:
    table = ChromaTrees
    green_model = demosaic_ast.load_ast("SELECTED_GREEN_MODELS_12-30/3331/model_ast")
    green_model_id = 5
    models = [ChromaSeedModel1(2, 12, True, green_model, green_model_id),\
              ChromaSeedModel2(2, 12, True, green_model, green_model_id), \
              ChromaSeedModel3(3, 12, True, green_model, green_model_id)]
    psnrs = [31.95, 32.05, 32.16]

  db.create_tables([table])

  tables = db.get_tables()
  print(tables)

  for i,model in enumerate(models):  
    key = i
    tree_hash = str(hash(model)).zfill(30)
    seed_record = table.create(model_id=key, tree_hash=tree_hash, machine="tefnut", \
                            experiment_dir="seed", tree_id_str=model.id_string(), psnr_0=psnrs[i], psnr_1=-1, psnr_2=-1)
    seed_record.save()
  
  query = table.select()
  for t in query:
    print(t.model_id, t.add_date, t.tree_hash, t.tree_id_str, t.machine, t.experiment_dir, t.psnr_0)


def select(password, tablename):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  if tablename == "testgreen":
    table = TestGreenTrees
  elif tablename == "green":
    table = GreenTrees
  else:
    table = ChromaTrees

  query = table.select()
  print(len(query))
  for t in query:
    print(t.model_id, t.tree_hash, t.machine, t.experiment_dir, t.psnr_0, t.psnr_1, t.psnr_2)


def select_range(password, tablename, id_min, id_max):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  if tablename == "testgreen":
    table = TestGreenTrees
  elif tablename == "green":
    table = GreenTrees
  else:
    table = ChromaTrees

  query = table.select().where(table.model_id < id_max, table.model_id > id_min)
  print(len(query))
  for t in query:
    print(t.model_id, t.tree_hash, t.machine, t.experiment_dir, t.psnr_0, t.psnr_1, t.psnr_2)



def find(password, tablename, tree_hash, tree_id_string, logger):

  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  if tablename == "testgreen":
    table = TestGreenTrees
  elif tablename == "green":
    table = GreenTrees
  else:
    table = ChromaTrees

  tree_hash = str(tree_hash).zfill(30)
  query = table.select().where(table.tree_hash == tree_hash, \
                                   table.tree_id_str == tree_id_string)
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


def mysql_insert(password, tablename, model_id, machine, exp_dir, tree_hash, id_str, psnrs, logger):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  if tablename == "testgreen":
    table = TestGreenTrees
  elif tablename == "green":
    table = GreenTrees
  else:
    table = ChromaTrees

  tree_hash = str(tree_hash).zfill(30)

  already_in_db = find(password, tablename, tree_hash, id_str, logger)
  if already_in_db is None:
    record = table.create(model_id=model_id, tree_hash=tree_hash, machine=machine, \
                              experiment_dir=exp_dir, tree_id_str=id_str, \
                              psnr_0=psnrs[0], psnr_1=psnrs[1], psnr_2=psnrs[2])
    record.save()

  else:
    logger.info(f"other machine also generated tree with model {model_id}'s hash {tree_hash}")


def mysql_delete_all(password, tablename):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()


  if tablename == "testgreen":
    table = TestGreenTrees
  elif tablename == "green":
    table = GreenTrees
  else:
    table = ChromaTrees

  found = table.select().where(table.model_id != 0)
  for f in found:
    f.delete_instance()


def mysql_delete(password, tablename, id_min, id_max):
  db_host = 'mysql.csail.mit.edu'
  db_name = 'ModelSearch'
  db_user = 'karima'
  db_password = password
  db_charset = 'utf8mb4'
    
  db_conn = {
      'host': db_host,
      'user': db_user,
      'passwd': db_password,
      'port': 3306,
  }

  db = MySQLDatabase(db_name, **db_conn)

  class BaseModel(Model):
    class Meta:
      database = db

  class GreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class ChromaTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  class TestGreenTrees(BaseModel):
    model_id = IntegerField(primary_key=True)
    machine = CharField(index=False, max_length=20)
    experiment_dir = CharField(index=False, max_length=40)
    tree_hash = CharField(index=True, max_length=30)
    tree_id_str = TextField()
    add_date = DateTimeField(default=datetime.datetime.now)
    psnr_0 = FloatField()
    psnr_1 = FloatField()
    psnr_2 = FloatField()

  if tablename == "testgreen":
    table = TestGreenTrees
  elif tablename == "green":
    table = GreenTrees
  else:
    table = ChromaTrees

  found = table.select().where(table.model_id >= id_min, table.model_id <= id_max)
  for t in found:
    print(t.model_id, t.tree_hash, t.machine, t.experiment_dir, t.psnr_0, t.psnr_1, t.psnr_2)
    t.delete_instance()


if __name__ == "__main__":
  # print("inserting seed tree")
  parser = argparse.ArgumentParser()
  parser.add_argument("--password", type=str)
  parser.add_argument("--table", type=str)
  parser.add_argument("--create", action='store_true')
  parser.add_argument("--delete_range", type=str, help="ids to delete from table")

  args = parser.parse_args()

  if args.delete_range:
    delete_range = [int(x.strip()) for x in args.delete_range.split(",")]
    print(f"deleting ids between {delete_range}")
    mysql_delete(args.password, args.table, delete_range[0], delete_range[1])
  if args.create:
    drop_table(args.password, args.table)
    create_table(args.password, args.table)

  print("checking database action worked...")
  select(args.password, args.table)




  
