from score.methods.schic_topic_model.scripts.data_conversion.topic_modeling_utlis import *
import os,sys
import glob
import pandas as pd
from tqdm import tqdm
import scipy.misc

from multiprocessing import Pool
from itertools import combinations


def chr_to_int_hg19(c):
	if c == 'chrX':
		return 23
	elif c == 'chrY':
		return 24
	elif c == 'chrM':
		return 25
	else:
		return int(c.replace('chr', ''))


def write_cell(filename, cell_number, resolution, set_lp_list, LP_list_saving, outDir, dist, anchor_list):
	mat_file = open(filename)
	cell_save = []
	for line_i, line in enumerate(mat_file):
		temp = line.strip()
		target = line.split()
		target[0] = chr_to_int_hg19(target[0])
		target[2] = chr_to_int_hg19(target[2])
		if target[0] == target[2] :
			lp = [target[0], int(target[1]), target[2], int(target[3])]
			if 'kim' in filename or 'synthetic' in filename or 'islet_all' in filename or 'ramani' in filename or 'li2019' in filename or 'simulated' in filename or 'hippocampus' in filename:
				search_lp = str(target[0]) + ":" + str(int(int(target[1]) + resolution / 2)) + "-" + str(int(int(target[3])+ resolution / 2))
				#print('kim', search_lp)
			elif anchor_list is None:
				search_lp = str(target[0]) + ":" + str(int(int(target[1]) - 1 + resolution / 2)) + "-" + str(int(int(target[3]) - 1 + resolution / 2))
			else:
				search_lp = str(target[0]) + ":" + str(int(int(target[1]))) + "-" + str(int(int(target[3])))
			#print(search_lp)
			if search_lp in set_lp_list:
				new_row = [cell_number, LP_list_saving.index(search_lp), int(target[4])]
				if new_row[-1] >= 0:
					if len(new_row) == 3:
						cell_save.append([cell_number, LP_list_saving.index(search_lp), int(target[4])])
					else:
						print(new_row, "not correct format...")
	mat_file.close()
	cell_save = np.asarray(cell_save)
	#print(cell_save)
	#cell_save = cell_save[cell_save[:,1].argsort()]

	out_mat_name = os.path.join(outDir, os.path.split(filename)[1][:-8] + ".sparse.matrix_" + str(dist))
	np.savetxt(out_mat_name, cell_save, fmt='%d', delimiter="\t", newline="\n")

def interaction_to_sparse(fileDir, outDir, libname, resolution, dist, anchor_file=None):
	if not os.path.exists(outDir):
		os.makedirs(outDir)

	assembly = 'hg19'
	chr_list = getChrs(assembly)
	valid_LPs = []

	if anchor_file is None:
		for this_chr in chr_list:
			chrs, midPoints = generateChrMidpoints(chrs = [this_chr], assembly = assembly, resolution = resolution)
			#print(chrs)
			#print(midPoints)
			offset = 0
			for mid_point in midPoints:
				for k in range(dist):
					if (offset+k) < len(midPoints):
						valid_LPs.append([int(this_chr), int(mid_point), int(this_chr), int(midPoints[k+offset])])
				offset += 1
	else:
		try:
			anchor_list = pd.read_csv(anchor_file, sep='\t', names=['chr', 'start', 'end', 'a1', 'length'], usecols=['chr', 'start', 'end', 'a1'], engine='python')  # read anchor list file
		except ValueError as e:
			anchor_list = pd.read_csv(anchor_file, sep='\t', names=['chr', 'start', 'end', 'a1', 'length', '?'], usecols=['chr', 'start', 'end', 'a1'], engine='python')  # read anchor list file
		chr_names = anchor_list['chr'].unique()
		for chr_name in chr_names:
			chr_anchors = anchor_list.loc[anchor_list['chr'] == chr_name]
			for (i, row1), (j, row2) in tqdm(combinations(chr_anchors.iterrows(), 2), total=scipy.special.comb(len(chr_anchors),2)):
				if abs(int(row1['start']) - int(row2['start'])) < (dist * resolution):
					valid_LPs.append([chr_to_int_hg19(row1['chr']), int(int(row1['start']) + resolution / 2), chr_to_int_hg19(row2['chr']), int(int(row2['start']) + resolution / 2)])

	LP_list_saving = [str(LP[0]) + ':' + str(LP[1]) + '-' + str(LP[3]) for LP in valid_LPs]
	print('Number of locus-pairs (LPs) to model:', len(LP_list_saving))
	set_lp_list = set(LP_list_saving)
	out_LP_name = os.path.join(outDir, libname + "_" + str(dist) + "_" + str(resolution) + "_LPnames.txt")
	np.savetxt(out_LP_name, LP_list_saving, fmt='%s', delimiter="\t", newline="\n")

	filenames = glob.glob(fileDir + '/*int.bed')
	#list_files = os.popen("find -type f -name " + fileDir + '/*int.bed').read()
	#list_files = os.popen('ls ' + fileDir + '/*int.bed').read()
	#filenames = list_files.split('\n')
	#filenames = filenames[:-1]
	num_cells = len(filenames)

	cell_number = 0
	res = []
	with Pool(12) as pool:
		for filename in sorted(filenames):
			res.append(pool.apply_async(write_cell, args=(filename, cell_number, resolution, set_lp_list, LP_list_saving, outDir, dist, anchor_file)))
			cell_number += 1
		for r in tqdm(res):
			try:
				r.get(timeout=1200)
			except Exception as e:
				print(e)

def main(argv):
	fileDir = sys.argv[1]
	outDir = sys.argv[2]
	libname = sys.argv[3]
	resolution = int(sys.argv[4])
	dist = int(sys.argv[5]) # number of bins
	anchor_file = None
	if len(sys.argv) > 6:
		anchor_file = sys.argv[6]

	interaction_to_sparse(fileDir, outDir, libname, resolution, dist, anchor_file)

	

if __name__ == "__main__":
	main(sys.argv)
