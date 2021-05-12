import demosaic_ast
from demosaic_ast import get_green_model_id, set_green_model_id

def insert_green_model(new_model_ast, green_model_asts, green_model_weights, green_model=None, green_model_weight_file=None):
  if green_model is None:
    green_model_id = get_green_model_id(new_model_ast)
    green_model_ast_file = green_model_asts[green_model_id]
    green_model_weight_file = green_model_weights[green_model_id]

    green_model = demosaic_ast.load_ast(green_model_ast_file)

  nodes = new_model_ast.preorder()
  for n in nodes:
    if type(n) is demosaic_ast.Input:
      if n.name == "Input(GreenExtractor)":
        n.node = green_model
        n.weight_file = green_model_weight_file
      elif hasattr(n, "node"): # other input ops my run submodels that also require the green model
        insert_green_model(n.node, green_model_asts, green_model_weights, green_model, green_model_weight_file)
