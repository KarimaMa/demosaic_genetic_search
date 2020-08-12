import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outfile", type=str)
parser.add_argument("--infile", type=str)
parser.add_argument("--subsets", type=int)
args = parser.parse_args()

lrs = [0 for s in range(args.subsets)]
with open(args.infile, "r") as in_f:
  for l in in_f:
    subset = int(l.split(' ')[-2])
    lr = float(l.split(' ')[-1])
    lrs[subset] = lr

with open(args.outfile, "w") as out_f:
  for lr in lrs:
    out_f.write(f"{lr}\n")

