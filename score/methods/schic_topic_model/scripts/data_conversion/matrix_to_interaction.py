from tqdm import tqdm
import sys,os

def matrix_to_interaction(inputfolder, mapbed, resolution, outputdir):
	print('Converting to locus pair interaction list...')
	mapbedfh = open(mapbed)

	bin_strs = False  # need to include 'bin' in token id
	anchor_strs = False
	if 'synthetic' in inputfolder or 'islet' in inputfolder or 'simulated' in inputfolder or 'hippocampus' in inputfolder:
		bin_strs = True
	if 'pfc' in inputfolder or 'kim' in inputfolder or 'ramani' in inputfolder or 'li2019' in inputfolder:
		anchor_strs = True
	os.makedirs(outputdir, exist_ok=True)

	bintocoord = {}
	chr_offset = 0
	prev_bin = 0
	chr_name = None
	for line in mapbedfh :
		tokens = line.split()
		if bin_strs:
			bin = int(tokens[3].replace('bin', '').replace('_', '')) - 1

			if chr_name is None:
				chr_name = tokens[0]
			# elif chr_name != tokens[0]:
			# 	chr_name = tokens[0]
			# 	chr_offset += prev_bin
			bintocoord[bin] = (tokens[0],str(int(tokens[1])+resolution/2))
			prev_bin = bin
		elif anchor_strs:
			bin = int(tokens[3].replace('A_', ''))
			bintocoord[bin] = (tokens[0],str(int(int(tokens[1]) + resolution/2)))
		else:
			bintocoord[int(tokens[3])] = (tokens[0],str(int(tokens[1])+resolution/2))


	list = os.popen('ls '+ inputfolder + '/*matrix').read()
	filenames = list.split('\n')[:-1]

	for matrixfilename in tqdm(filenames) :
		if matrixfilename != '' :
			matrixfilefh = open(matrixfilename)
			outfilefh = open(os.path.join(outputdir, os.path.split(matrixfilename)[1][:-7]+'.int.bed'),'w')
			#print(matrixfilename)
			for line in matrixfilefh :
				tokens = line.split()
				bin1_idx = int(tokens[0])
				bin2_idx = int(tokens[1])
				
				#print(bintocoord.keys())
				#print(tokens[0], tokens[1])
				try:
					firstcoor = bintocoord[bin1_idx]
					secondcoor = bintocoord[bin2_idx]
					outtokens = [firstcoor[0], str(int(firstcoor[1].split('.')[0])), secondcoor[0], str(int(secondcoor[1].split('.')[0])), str(int(tokens[2].split('.')[0])), tokens[3]]
					outtokens = [str(t) for t in outtokens]
					outline = '\t'.join(outtokens)
					print(outline, file=outfilefh)
				except KeyError as e:
					pass


			outfilefh.close()
