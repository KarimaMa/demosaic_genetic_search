from model_lib import *
from torch_model import ast_to_model


if __name__ == "__main__":
	multires = multires_green_model()
	print(multires.dump())
	multires.save_ast("foo.data")
	new_multires = load_ast("foo.data")
	print(new_multires.dump())

	model = new_multires.ast_to_model()
