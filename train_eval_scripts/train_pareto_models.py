import csv
import argparse
import sys
import os
sys.path.append(sys.path[0].split("/")[0])
sys.path.append(os.path.dirname(sys.path[0]) + "/sys_run")
import util
from train import train_model
import demosaic_ast
import torch_model
import logging

parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--lr_search_steps', type=int, default=2, help='how many line search iters for finding learning rate')
parser.add_argument('--variance_min', type=float, default=0.003, help='minimum validation psnr variance')
parser.add_argument('--variance_max', type=float, default=0.02, help='maximum validation psnr variance')
parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='training report frequency')
parser.add_argument('--save_freq', type=float, default=2000, help='trained weights save frequency')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--model_initializations', type=int, default=3, help='number of weight initializations to train per model')
parser.add_argument('--train_portion', type=int, default=1e5, help='portion of training data to use')
parser.add_argument('--validation_freq', type=int, default=50, help='validation frequency for assessing validation PSNR variance')

parser.add_argument('--pareto_models', type=str)
parser.add_argument('--input_model_dir', type=str)
parser.add_argument('--output_model_dir', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--training_file', type=str, help='filename of file with list of training data image files')
parser.add_argument('--validation_file', type=str, help='filename of file with list of validation data image files')

args = parser.parse_args()


def launch_train(args, model_ast, model_dir, model_id):

	log_format = '%(asctime)s %(levelname)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, \
	      format=log_format, datefmt='%m/%d %I:%M:%S %p')

	models = [model_ast.ast_to_model() for i in range(args.model_initializations)]

	training_logger = util.create_logger(f'model_{model_id}_train_logger', logging.INFO, log_format, \
											os.path.join(model_dir, f'model_{model_id}_training_log'))
	training_logger.info("args = %s", args)

	val_psnrs, train_psnrs = train_model(args, args.gpu, model_id, models, model_dir, training_logger)

	return val_psnrs, train_psnrs


def load_model_ast(input_model_dir, model_id):
	model_info_file = os.path.join(input_model_dir, f"{model_id}/model_info")
	model_ast = util.load_ast_from_file(model_info_file)
	return model_ast


with open(args.pareto_models, newline='\n') as csvf:
	reader = csv.DictReader(csvf, delimiter=',')
	for r in reader:
		model_id = r['model_id']
		compute_cost = r['compute_cost']
		psnr = r['psnr']

		model_ast = load_model_ast(args.input_model_dir, model_id)
		print(model_ast.dump())

		model_dir = os.path.join(args.output_model_dir, f"{model_id}")
		print(f"output model dir {model_dir}")
		print(f"prev psnr {psnr} compute_cost {compute_cost}")
		util.create_dir(model_dir)

		val_psnrs, train_psnrs = launch_train(args, model_ast, model_dir, model_id)





	