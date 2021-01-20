from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
import re
import Bio
import Bio.SeqIO
import scipy.stats as scistat
import argparse
import glob

"""Find the statistical significance of the binding strength at each regulatory DNA 
by comparing the TF's foreground Calling cards data with the no-TF control data.

Example:
python3 HELPER_SCRIPTS/find_sig_promoters.py -o RESOURCES/Yeast_CallingCards/tss1000to500b_sig -g RESOURCES/Yeast_CallingCards/gnashy/ -b RESOURCES/Yeast_CallingCards/gnashy/NOTF_Control.gnashy -p RESOURCES/Yeast_genome/S288C_R64_sacCer3_deGoer2020_tss1000to500b.bed
"""


def parse_args(argv):
	parser = argparse.ArgumentParser(description='This is the find_sig_promoters module for yeast.')
	parser.add_argument(
		'-o', '--outputpath', 
		help='output path', required=True)
	parser.add_argument(
		'-g', '--gnashypath', 
		help='gnashy file path', required=True)
	parser.add_argument(
		'-b', '--bgfile', 
		help='background hop distribution path and filename', required=True)
	parser.add_argument(
		'-p', '--promfile', 
		help='promoter bed file', required=True)
	parsed = parser.parse_args(argv[1:])
	return parsed


def find_significant_IGRs(outputpath, experiment_gnashy_filename, background_gnashy_filename, orfPromoter_filename):
	#read in promoter regions and populate columns
	IGR_frame = readin_promoters(orfPromoter_filename)

	#read in background and experiment hops gnashy files
	#and populate expected and observed hops
	[IGR_frame,bg_hops,exp_hops] = readin_hops(IGR_frame,background_gnashy_filename,experiment_gnashy_filename)
	
	#read in orf table and populate common names
	# IGR_frame = populate_common_names(IGR_frame,orfcoding_filename)

	#compute cumulative hypergeometric
	# IGR_frame = compute_cumulative_hypergeometric(IGR_frame,bg_hops,exp_hops)

	#compute poisson
	IGR_frame = compute_cumulative_poisson(IGR_frame,bg_hops,exp_hops)
	
	#remove His3 false positive
	# IGR_frame = IGR_frame[IGR_frame['Right Common Name'] != "HIS3"]

	#output frame
	IGR_frame = IGR_frame.sort_values(["Systematic Name"],ascending = [True])
	experiment_gnashy_basename = os.path.basename(experiment_gnashy_filename).strip("gnashy")
	output_filename = outputpath + experiment_gnashy_basename +'sig_prom.txt'
	IGR_frame.to_csv(output_filename,sep = '\t',index = None)


def readin_promoters(filename):
	#initialize IGR Dataframe
	IGR_frame = pd.DataFrame(columns = ["Systematic Name", "Common Name",
										"Chr", "Start", "Stop", 
										"Background Hops", 
										"Experiment Hops",
										"Background TPH",
										"Experiment TPH", 
										"Experiment TPH BS", 
										"Poisson pvalue", 
										"Gene Body Background Hops", 
										"Gene Body Experiment Hops", 
										"Gene Body Poisson pvalue"])
	## read in promoter bed file
	prom_frame = pd.read_csv(filename, delimiter = '\t')
	prom_frame.columns = ['chr', 'start', 'stop', 'name', 'score', 'strand']
	prom_frame["chr"] = [prom_frame["chr"][i].strip('chr') for i in range(len(prom_frame))]
	## fill the dataframe
	IGR_frame["Systematic Name"] = prom_frame["name"]
	IGR_frame["Chr"] = prom_frame["chr"].astype("|S10")
	IGR_frame["Start"] = prom_frame["start"]
	IGR_frame["Stop"] = prom_frame["stop"]
	return IGR_frame


def readin_hops(IGR_frame,background_gnashy_filename,experiment_gnashy_filename):
	## read in the 3-column background and experiment gnashy data
	background_frame = pd.read_csv(
		background_gnashy_filename, delimiter="\t", header=None, usecols=[0, 1, 4])
	background_frame.columns = ['Chr','Pos','Reads']
	bg_hops = len(background_frame)
	experiment_frame = pd.read_csv(
		experiment_gnashy_filename, delimiter="\t", header=None, usecols=[0, 1, 4])
	experiment_frame.columns = ['Chr','Pos','Reads']
	exp_hops = len(experiment_frame)
	## force chromosome in gnashy files to be string
	background_frame["Chr"] = background_frame["Chr"].astype("|S10")
	experiment_frame["Chr"] = experiment_frame["Chr"].astype("|S10")
	## iter through each row and fill the dataframe
	for indexvar in IGR_frame.index:
		pos = [IGR_frame.ix[indexvar,"Start"], IGR_frame.ix[indexvar,"Stop"]]
		IGR_frame.ix[indexvar,"Background Hops"] = len(background_frame[(background_frame["Chr"]==IGR_frame.ix[indexvar,"Chr"]) & (background_frame["Pos"] <= max(pos)) &(background_frame["Pos"] >= min(pos))])
		IGR_frame.ix[indexvar,"Experiment Hops"] = len(experiment_frame[(experiment_frame["Chr"]==IGR_frame.ix[indexvar,"Chr"]) & (experiment_frame["Pos"] <= max(pos)) &(experiment_frame["Pos"] >= min(pos))])
		IGR_frame.ix[indexvar,"Background TPH"] = float(IGR_frame.ix[indexvar,"Background Hops"])/float(bg_hops) *100000
		IGR_frame.ix[indexvar,"Experiment TPH"] = float(IGR_frame.ix[indexvar,"Experiment Hops"])/float(exp_hops) *100000
		IGR_frame.ix[indexvar,"Experiment TPH BS"] = IGR_frame.ix[indexvar,"Experiment TPH"] - IGR_frame.ix[indexvar,"Background TPH"]
		if IGR_frame.ix[indexvar,"Experiment TPH BS"] < 0:
			IGR_frame.ix[indexvar,"Experiment TPH BS"] = 0
	return IGR_frame,bg_hops,exp_hops


def populate_common_names(IGR_frame,orfcoding_filename):
	orf_dict = {}
	for x in Bio.SeqIO.parse(orfcoding_filename,"fasta"):
		pattern = '(?<=Y)\S+(?= )'
		matchobj = re.search(pattern,x.description)
		if matchobj:
			y_name = "Y"+matchobj.group(0)
			pattern = '\S+(?= SGDID)'
			matchobj = re.search(pattern,x.description)
			if matchobj:
				orf_dict[y_name] = matchobj.group(0)
	for idx,row in IGR_frame.iterrows():
		if IGR_frame.ix[idx,"Left Feature"] in orf_dict:
			IGR_frame.ix[idx,"Left Common Name"] = orf_dict[row["Left Feature"]]
		else:
			IGR_frame.ix[idx,"Left Common Name"] = row["Left Feature"]
		if IGR_frame.ix[idx,"Right Feature"] in orf_dict:
			IGR_frame.ix[idx,"Right Common Name"] = orf_dict[row["Right Feature"]]
		else:
			IGR_frame.ix[idx,"Right Common Name"] = row["Right Feature"]
	return IGR_frame


def compute_cumulative_hypergeometric(IGR_frame,bg_hops,exp_hops):
	#usage
	#scistat.hypergeom.cdf(x,M,n,N)
	#where x is observed number of type I events (white balls in draw) (experiment hops at locus)
	#M is total number of balls (total number of hops)
	#n is total number of white balls (total number of expeirment hops)
	#N is the number of balls drawn (total hops at a locus)
	for idx,row in IGR_frame.iterrows():
		IGR_frame.ix[idx,"CHG pvalue"] = 1-scistat.hypergeom.cdf(IGR_frame.ix[idx,"Experiment Hops"]-1,(bg_hops + exp_hops),exp_hops,(IGR_frame.ix[idx,"Experiment Hops"]+IGR_frame.ix[idx,"Background Hops"]))
	return IGR_frame


def compute_cumulative_poisson(IGR_frame,bg_hops,exp_hops):
	#usage
	#scistat.poisson.cdf(x,mu)
	pseudocount = 0.2
	for idx,row in IGR_frame.iterrows():
		IGR_frame.ix[idx,"Poisson pvalue"] = 1-scistat.poisson.cdf((IGR_frame.ix[idx,"Experiment Hops"]+pseudocount),(IGR_frame.ix[idx,"Background Hops"] * (float(exp_hops)/float(bg_hops)) + pseudocount))
	return IGR_frame


def main(argv):
	parsed = parse_args(argv)
	parsed.gnashypath += "/" if not parsed.gnashypath.endswith("/") else ""
	parsed.outputpath += "/" if not parsed.outputpath.endswith("/") else ""
	
	for gnashyfile in glob.glob(parsed.gnashypath+"*.gnashy"):
		if os.path.normpath(parsed.bgfile) != os.path.normpath(gnashyfile): 
			print("... working on", gnashyfile)
			find_significant_IGRs(parsed.outputpath, gnashyfile, parsed.bgfile, parsed.promfile)


if __name__ == '__main__': 
	main(sys.argv)



