fname ="MODEL_SEARCH_10_16_12PM/model_database/ModelDatabase-snapshot-20201016-115418"
outname = "MODEL_SEARCH_10_16_12PM/model_database/ModelDatabase-snapshot-20201016-115418v2"

with open(outname, "w") as wf:
  for i,l in enumerate(open(fname, "r")):
    if i == 0:
      wf.write(l)
      continue
    data = l.split('"')
    data[1] = data[1].replace(',',';')
    out = data[0] + data[1] + data[2]
    wf.write(out)  
