from peewee import *
from tree import *
from model_lib import *
import datetime

db_host = 'mysql.csail.mit.edu'
db_name = 'DemosaicSearch'
db_user = 'karima'
db_password = 'imsvalley7'
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

class Tree(BaseModel):
  model_id = IntegerField(primary_key=True)
  tree_hash = CharField(index=True, max_length=30)
  add_date = DateTimeField(default=datetime.datetime.now)

db.create_tables([Tree])

tables = db.get_tables()
print(tables)

timestamp = datetime.datetime.now()
print(timestamp)

# get next available primary key
trees = Tree.select()
key = max([t.model_id for t in trees])
print(f"starting key {key}")

basic1d = basic1D_green_model()
basic2d = basic2D_green_model()
multires1d = multires_green_model()

key += 1
tree_hash = str(hash(basic1d)).zfill(30)
basic1d_record = Tree.create(model_id=key, tree_hash=tree_hash)
#basic1d_record.tree_hash = tree_hash
basic1d_record.save()

key += 1
tree_hash = str(hash(basic2d)).zfill(30)
basic2d_record = Tree.create(model_id=key, tree_hash=tree_hash)
#basic2d_record.tree_hash = tree_hash
basic2d_record.save()
query = Tree.select()

key += 1
tree_hash = str(hash(multires1d)).zfill(30)
multires1d_record = Tree.create(model_id=key, tree_hash=tree_hash)
#multires1d_record.tree_hash = tree_hash
multires1d_record.save()

query = Tree.select()
for t in query:
  print(t.model_id, t.add_date, t.tree_hash)

print("trees added before")
query = Tree.select().where(Tree.add_date < timestamp)
for t in query:
  print(t.model_id, t.add_date, t.tree_hash)

