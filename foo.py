def f(x=None):
  if x is None:
    x = []
  x += [1]
  return x

print(f())
print(f())
print(f())
