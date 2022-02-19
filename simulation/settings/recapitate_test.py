import numpy as np
import pyslim, tskit, msprime
import sys

SEED=5 #CONTROL
OUTFILE='test_vcf_after_recap_sweep.vcf' 
np.random.seed(SEED)


orig_ts = tskit.load("example_sweep_sim.trees") #name of trees file from slim

#recapitate
rts = pyslim.recapitate(orig_ts, recombination_rate=1e-8, ancestral_Ne=10000)

#simplify
alive_inds = rts.individuals_alive_at(0)
keep_indivs = np.random.choice(alive_inds, 64, replace=False)
keep_nodes = []
for i in keep_indivs:
	keep_nodes.extend(rts.individual(i).nodes)
sts = rts.simplify(keep_nodes, keep_input_roots=True)

#add neutral mutations
ts = pyslim.SlimTreeSequence(msprime.sim_mutations(
								sts, rate=1.5e-8, 
								model=msprime.SLiMMutationModel(type=0),
								keep=True))

#output to VCF
indivlist = []
for i in ts.individuals_alive_at(0):
	ind = ts.individual(i)
	if ts.node(ind.nodes[0]).is_sample():
		indivlist.append(i)
		assert ts.node(ind.nodes[1]).is_sample()
with open(OUTFILE, 'w') as vcffile:
	ts.write_vcf(vcffile, individuals=indivlist)

##WANT TO DELETE .trees FILE