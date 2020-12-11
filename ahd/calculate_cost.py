"""
calculates cost of ahd 
"""

import math
import argparse

LOGEXP_COST = 10
POW_COST = 2*LOGEXP_COST + 1 
DIV_COST = 10
ADDMUL_COST = 1
SUB_COST = 1
COMP_COST = 1
SELECT_COST = 1
ABSVAL_COST = 1
MAX_COST = 1
MIN_COST = 1
CEIL_COST = 1
SQRT_COST = 10
BITWISEOP_COST = 1

MEDIAN4_COST = 10
MEDIAN8_COST = 30

NUM_PIXELS = 128*128

COUNT_MBALL = False


def median_cost(n):
	return n * MEDIAN_FILTER_COST

def norm_cost(n):
	cost = n * ADDMUL_COST + SQRT_COST
	return cost

def hinterp_cost(green_only):
	green_cost = 5 * ADDMUL_COST
	red_cost = 9 * ADDMUL_COST
	blue_cost = 9 * ADDMUL_COST
	if green_only:
		return green_cost
	return green_cost + red_cost + blue_cost

def vinterp_cost(green_only):
	return hinterp_cost(green_only)

"""
cost of converting RGB to LAB
"""
def lab_conversion_cost():
	matmul_cost = 9
	f_cost = POW_COST + 2 * ADDMUL_COST + 2 * COMP_COST # cost of function f

	L_cost = f_cost + DIV_COST + ADDMUL_COST + SUB_COST
	a_cost = 2 * (f_cost + DIV_COST) + ADDMUL_COST + SUB_COST
	b_cost = a_cost
	return matmul_cost + L_cost + a_cost + b_cost

def epsilon_cost():
	Lgradient_cost = SUB_COST + ABSVAL_COST # finite diff
	epsilonL_cost = 4 * Lgradient_cost + 2 * MAX_COST + MIN_COST # take min of two maxes of two gradients

	Cgradient_cost = 2 * SUB_COST + 2 * POW_COST + ADDMUL_COST # squared norm of finite diff in AB space
	epsilonC_cost = 4 * Cgradient_cost + 2 * MAX_COST + MIN_COST
	epsilonC_cost += POW_COST # do sqrt because we computed the min max of a squared norm

	return epsilonL_cost + epsilonC_cost

"""
should this be considered as a precomputed filter?
computes the locations in a k x k window centered at pixel x
that are <= delta distance away from x
"""
def mball_cost(delta):
	w = delta * 2 + 1
	H_size = w * w 
	cost = H_size * (norm_cost(2) + COMP_COST + 4 * ADDMUL_COST)
	print(cost)
	return cost

"""
for each point x in the image, computes how many points within a 
delta radius ball centered at x are within luminance epsilon and
chroma epsilon of the luminance and chroma value at x
"""
def homogeneity_cost(delta):
	# consider removing mball cost because it can be precomputed 
	if COUNT_MBALL:
		cost = (CEIL_COST + mball_cost(delta) + POW_COST) / NUM_PIXELS 
	else:
		cost = (CEIL_COST + POW_COST) / NUM_PIXELS

	window_w = delta * 2 + 1
	n_ball_points = (math.pi / 4) * window_w**2 # ratio of circle area to square area

	# Ccost = 2 * SUB_COST + 2 * POW_COST + ADDMUL_COST + COMP_COST  # computes norm of difference in color (AB) space
	# Ucost = BITWISEOP_COST # bitwise AND of L and C
	# Kcost = ADDMUL_COST # adds U to running sum of ball points surrounding x within homogeneity limits 

	#cost += n_ball_points * (Lcost + Ccost + Ucost + Kcost)
	Lcost = (SUB_COST + ABSVAL_COST + COMP_COST) * n_ball_points # compares difference in luminance channel
	AB_cost = n_ball_points * 4 + 13 # calculated using Andrew's formuala 
	cost += Lcost + AB_cost
	return cost

def artifact_cost(iterations, green_only):	
	r_cost = SUB_COST + MEDIAN8_COST + ADDMUL_COST # median of R - G, then add back G
	b_cost = SUB_COST + MEDIAN8_COST + ADDMUL_COST # median of B - G, then add back G 
	gr_cost = SUB_COST + MEDIAN4_COST + ADDMUL_COST # median of G - R, then add back R
	gb_cost = SUB_COST + MEDIAN4_COST + ADDMUL_COST # median of G - B, then add back B
	g_cost = ADDMUL_COST + DIV_COST # average G adjusted by R and B

	if green_only:
		cost = iterations * (gr_cost + gb_cost + g_cost)
	else:
		cost = iterations * (r_cost + b_cost + gr_cost + gb_cost + g_cost)
	return cost

def compute_cost(delta, iterations, green_only, LAB):
	cost = hinterp_cost(green_only)
	print(f"h interp cost {hinterp_cost(green_only)}")
	cost += vinterp_cost(green_only)
	print(f"v interp cost {vinterp_cost(green_only)}")

	if LAB:
		cost += 2*lab_conversion_cost() # LAB conversion of h and v images
		print(f"lab_conversion_cost {2*lab_conversion_cost()}")

	cost += epsilon_cost()
	print(f"epsilon_cost {epsilon_cost()}")

	cost += 2 * homogeneity_cost(delta) # Homogeneity maps of h and v images
	print(f"homogeneity_cost {homogeneity_cost(delta):.2f}")

	average_hmap_cost = 9 * ADDMUL_COST # box blur of homogeneity maps
	average_vmap_cost = average_hmap_cost

	cost += average_hmap_cost + average_vmap_cost

	homogeneity_comp_cost = COMP_COST + SELECT_COST # set output to vertical interp if vmap > hmap

	cost += homogeneity_comp_cost
	cost += artifact_cost(iterations, green_only)
	print(f"artifact_cost {artifact_cost(iterations, green_only)}")

	return cost



parser = argparse.ArgumentParser()
parser.add_argument("--delta", type=int)
parser.add_argument("--iters", type=int)
parser.add_argument("--green_only", action="store_true")
parser.add_argument("--LAB", action="store_true")
args = parser.parse_args()


print(f"ahd compute cost with delta={args.delta} iterations={args.iters} {compute_cost(args.delta, args.iters, args.green_only, args.LAB):.2f}")



