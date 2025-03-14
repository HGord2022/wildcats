import msprime
import tskit
import numpy as np
import pyslim
import subprocess
import os
from collections import namedtuple
import logging
import shlex
import re
from tabulate import tabulate
from priors import priors
import time
import utils
import pickle

"""
This script defines a simulation model for wildcat populations using the msprime, tskit, and pyslim libraries. 
It includes a class `WildcatModel` that encapsulates the simulation logic and several helper functions.

Classes:
    WildcatModel: 
        A class to simulate wildcat populations with various demographic events and parameters.
    GenotypeData: 
        A class to collate results from the simulation or real data for analysis with scikit-allel.

Functions:
    add_seed_suffix_to_file(filename, seed):
        Adds a suffix with the random seed to a filename to avoid clashes.
    tree_summary(tree_seq):
        Prints a summary of a tree sequence including the number of trees, coalesced trees, tree heights, 
        number of alive individuals, samples, populations, variants, and sequence length.
    get_sampled_nodes(tree_seq):
        Finds the sampled nodes from a simplified tree sequence and returns a namedtuple with the nodes 
        for each population.

Methods in WildcatModel:
    __init__(self, seq_length, recombination_rate, mutation_rate, decap_trees_filename="decap.trees", add_seed_suffix=True):
        Initializes the WildcatModel with the given parameters.
    simulate(self, bottleneck_strength_domestic, bottleneck_strength_wild, bottleneck_time_domestic, bottleneck_time_wild, 
             captive_time, div_time, mig_length_post_split, mig_rate_post_split, mig_length_wild, mig_rate_wild, 
             mig_rate_captive, pop_size_captive, pop_size_domestic_1, pop_size_domestic_2, pop_size_wild_1, 
             pop_size_wild_2, n_samples=[30, 30, 30], seed=None):
        Runs the simulation with the specified parameters and returns the tree sequence and time taken.
    slim_command(self, pop_size_domestic_1, pop_size_wild_1, pop_size_captive, mig_rate_captive, mig_length_wild, 
                 mig_rate_wild, captive_time, seed, slim_script_filename='slim_model.slim', add_seed_suffix=True):
        Constructs the command to run the SLiM simulation with the given parameters.
    run_slim(self, command):
        Runs the SLiM simulation from the command line and returns the decapitated tree sequence.
    get_demography(pop_size_domestic_1, pop_size_wild_1, pop_size_captive, pop_size_domestic_2, pop_size_wild_2, 
                   div_time, mig_rate_post_split, mig_length_post_split, bottleneck_time_wild, bottleneck_strength_wild, 
                   bottleneck_time_domestic, bottleneck_strength_domestic):
        Defines the demography model for recapitation, including bottlenecks, population size changes, and migration.
    recapitate(self, decap_trees, demography, seed, demography_debugger=False):
        Recapitates the tree sequence under the specified demography model, adds mutations, and returns the tree sequence.
    sample_nodes(self, tree_seq, sample_sizes, seed, concatenate=True):
        Samples nodes of individuals from the extant populations for simplification.

Methods in GenotypeData:
    __init__(self, tree_seq=None, callset=None, subpops=None, seq_length=None):
        Initializes the GenotypeData object with either a tree sequence or a callset.
    _initialize_from_tree_seq(self, tree_seq):
        Initializes the GenotypeData object from a tree sequence.
    _initialize_from_callset(self, callset):
        Initializes the GenotypeData object from a callset.
"""

class WildcatModel:

    def __init__(
        self,
        seq_length: int,
        recombination_rate: float,
        mutation_rate: float,
        decap_trees_filename: str = "decap.trees",
        add_seed_suffix: bool = True,
        ):
        """
        
        Wildcat model object. Recommended that the directory "../output/" is set up so
        the default output paths work.

        Args:
            seq_length (int): Length of sequence to simulate in base pairs.
            recombination_rate (float): Recombination rate per base pair.
            mutation_rate (float): Mutation rate per base pair.
            random_seed (Optional[int], optional): Random seed. Defaults to None.
            decap_trees_filename (str): File to write out the decapitated tree sequence from slim.
            add_seed_suffix (bool): Whether to append _{seed} before file extension to limit risk of file name clashes.
        """
        self.seq_length = seq_length
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.add_seed_suffix = add_seed_suffix
        self.decap_trees_filename = decap_trees_filename


    def simulate(
        self,
        bottleneck_strength_domestic: float,
        bottleneck_strength_wild: float,
        bottleneck_time_domestic: int,
        bottleneck_time_wild: int,
        captive_time: int,
        div_time: int,
        mig_length_post_split: int,
        mig_rate_post_split: float,
        mig_length_wild: int,
        mig_rate_wild: float,
        mig_rate_captive: float,
        pop_size_captive: int,
        pop_size_domestic_1: int,
        pop_size_domestic_2: int,
        pop_size_wild_1: int,
        pop_size_wild_2: int,
        n_samples=[30, 30, 30],
        seed=None
        ):
        """
        Args:
            bottleneck_strength_domestic: float
            bottleneck_strength_wild: float
            bottleneck_time_domestic: int
            bottleneck_time_wild: int
            captive_time: int
            div_time: int
            mig_length_post_split: int
            mig_rate_post_split: float
            mig_length_wild: int
            mig_rate_wild: float
            mig_rate_captive: float
            pop_size_captive: int
            pop_size_domestic_1: int
            pop_size_domestic_2: int
            pop_size_wild_1: int
            pop_size_wild_2: int
            n_samples (list, optional): Number of samples from each population (domestic, wild, captive). 
            Defaults to [30, 30, 30].
            seed (int, optional): Random seed. Defaults to None.
        """

        np.random.seed(seed)

        start = time.time() 

        # Run simulation
        command = self.slim_command(
            pop_size_domestic_1=int(pop_size_domestic_1),
            pop_size_wild_1=int(pop_size_wild_1),
            pop_size_captive=int(pop_size_captive),
            mig_rate_captive=mig_rate_captive,
            mig_length_wild=int(mig_length_wild),
            mig_rate_wild=mig_rate_wild,
            captive_time=int(captive_time),
            seed=seed
        )

        tree_seq = self.run_slim(command)  # Recent history (decapitated)


        demography = self.get_demography(
            pop_size_domestic_1=int(pop_size_domestic_1),
            pop_size_wild_1=int(pop_size_wild_1),
            pop_size_captive=int(pop_size_captive),
            pop_size_domestic_2=int(pop_size_domestic_2),
            pop_size_wild_2=int(pop_size_wild_2),
            div_time=int(div_time),
            mig_rate_post_split=mig_rate_post_split,
            mig_length_post_split=int(mig_length_post_split),
            bottleneck_time_wild=int(bottleneck_time_wild),
            bottleneck_strength_wild=bottleneck_strength_wild,
            bottleneck_time_domestic=int(bottleneck_time_domestic),
            bottleneck_strength_domestic=bottleneck_strength_domestic
        )

        tree_seq = self.recapitate(tree_seq, demography, seed)


        # Take samples to match number of samples in real WGS data
        samples = self.sample_nodes(tree_seq, n_samples, seed)

        # simplify tree sequence
        tree_seq = tree_seq.simplify(samples=samples)

        # apply minor allele count filter to match real genome data
        tree_seq = tree_seq.delete_sites(utils.mac_filter(tree_seq, count=3))

        filename_map = "./simulated_tree.pickle"

        with open(filename_map, 'wb') as handle:
            pickle.dump(tree_seq, handle, protocol=pickle.DEFAULT_PROTOCOL)

        end = time.time()
        print(tree_seq.num_sites)
        time_taken = end - start

        return tree_seq, time_taken

    def slim_command(
        self,
        pop_size_domestic_1,
        pop_size_wild_1,
        pop_size_captive,
        mig_rate_captive,
        mig_length_wild,
        mig_rate_wild,
        captive_time,
        seed,
        slim_script_filename='slim_model.slim',
        add_seed_suffix: bool = True,
        ):

        param_dict = {
            "pop_size_domestic_1": pop_size_domestic_1,
            "pop_size_wild_1": pop_size_wild_1,
            "pop_size_captive": pop_size_captive,
            "mig_rate_captive": mig_rate_captive,
            "mig_length_wild": mig_length_wild,
            "mig_rate_wild": mig_rate_wild,
            "captive_time": captive_time,
            "length": self.seq_length,
            "recombination_rate": self.recombination_rate
        }

        if add_seed_suffix:
            decap_trees_filename = add_seed_suffix_to_file(self.decap_trees_filename, seed)

        param_string = "".join([f"-d {k}={v} " for k,v in param_dict.items()])
        command = f"""slim {param_string} -d decap_trees_filename="{decap_trees_filename}" -s {seed} {slim_script_filename} """

        return command

    def run_slim(self, command):
        """Runs SLiM simulation from command line to get the decapitated tree sequence."""

        command_split = shlex.split(command, posix=False)

        if self.add_seed_suffix:
            seed = re.findall(r"%s(\d+)" % " -s ", command)[0]  # regex to find seed
            decap_trees_filename = add_seed_suffix_to_file(self.decap_trees_filename, seed)
        else:
            decap_trees_filename = self.decap_trees_filenames
        
        try:
            logging.info(f"Running command: {command}")
            subprocess.check_output(command_split)  # This may not work on windows!
            tree_seq = tskit.load(decap_trees_filename)

        except subprocess.CalledProcessError as err:
            print(f"Running slim failed with command {command}")
            raise err

        finally:  # Ensure file are cleaned up even if above fails
            os.remove(decap_trees_filename)

        return tree_seq

    @staticmethod
    def get_demography(
        pop_size_domestic_1, pop_size_wild_1, pop_size_captive, pop_size_domestic_2, pop_size_wild_2,
        div_time, mig_rate_post_split, mig_length_post_split, bottleneck_time_wild,
        bottleneck_strength_wild, bottleneck_time_domestic, bottleneck_strength_domestic):
        """Model for recapitation, including bottlenecks, population size changes and migration.
        Returns list of demographic events sorted in time order. Note that if parameters are drawn from priors this
        could have unexpected consequences on the demography. sim.utils.check_params() should mitigate this issue."""

        migration_time_2 = div_time-mig_length_post_split

        domestic, wild, captive = "p0", "p1", "p2"  # Names match slim for recapitation

        demography = msprime.Demography()
        demography.add_population(name=domestic, initial_size=pop_size_domestic_1)
        demography.add_population(name=wild, initial_size=pop_size_wild_1)
        demography.add_population(name=captive, initial_size=pop_size_captive, initially_active=False)
        demography.add_population(name="mrca", initial_size=pop_size_domestic_2+pop_size_wild_2, initially_active=False)

        demography.add_population_parameters_change(
            time=bottleneck_time_domestic, initial_size=pop_size_domestic_2, population=domestic
        )
        demography.add_instantaneous_bottleneck(
            time=bottleneck_time_domestic, strength=bottleneck_strength_domestic, population=domestic
        )

        demography.add_population_parameters_change(
            time=bottleneck_time_wild, initial_size=pop_size_wild_2, population=wild
        )

        demography.add_instantaneous_bottleneck(
            time=bottleneck_time_wild, strength=bottleneck_strength_wild, population=wild
        )

        demography.add_symmetric_migration_rate_change(
            time=migration_time_2, populations=[domestic, wild], rate=mig_rate_post_split
        )

        demography.add_population_split(time=div_time, derived=[domestic, wild], ancestral="mrca")
        demography.sort_events()
        return demography

    def recapitate(self, decap_trees, demography, seed: int, demography_debugger=False):
        """Recapitates tree sequence under model specified by demography.
        Adds mutations and sequencing errors. Returns tskit.tree_sequence."""
        
        tree_seq = pyslim.recapitate(
            decap_trees,
            recombination_rate=self.recombination_rate,
            demography=demography,
            random_seed=seed)

        # Overlay mutations
        tree_seq = msprime.mutate(
            tree_seq,
            rate=self.mutation_rate,
            random_seed=seed
            )

        tree_seq = tree_seq.simplify()

        if demography_debugger:
            dd = demography.debug()
            print("### Here is msprime population info ###")
            dd.print_history()

        
        return tree_seq

    def sample_nodes(self, tree_seq, sample_sizes, seed, concatenate=True):
        """
        Samples nodes of individuals from the extant populations, which can then be used for simplification.
        Warning: Node IDs are not kept consistent during simplify!.

        param: tree_seq, tskit.tree_sequence: tree sequence object
        param: sample_sizes, list: a list of length 3, with each element giving the sample size
        for the domestic, wildcat and captive cat population respectively.
        param: concatenate, bool: If False, samples for each population are kept in seperate items in the list.

        Returns
        ------------
        np.array of nodes. If concatenate is false, an array for each population is returned in a list.

        """

        # Have to use individuals alive at to avoid remembered individuals at slim/msprime interface
        nodes = np.array([tree_seq.individual(i).nodes for i in pyslim.individuals_alive_at(tree_seq, 0)])
        pop = np.array([tree_seq.individual(i).population for i in pyslim.individuals_alive_at(tree_seq, 0)])
        np.random.seed(seed)
        samples = []
        for pop_num in range(3):
            sampled_inds = np.random.choice(
                np.where(pop == pop_num)[0], replace=False, size=sample_sizes[pop_num]
                )

            sampled_nodes = nodes[sampled_inds].ravel()
            samples.append(sampled_nodes)

        samples = np.concatenate(samples) if concatenate else samples
        return samples


def get_sampled_nodes(tree_seq):
    """
    Finds the sampled nodes from a simplified tree_sequence

    returns: namedtuple, where names are the population, and each tuple element is
    a numpy array of length 2 lists (the nodes from an individual)
    """
    Nodes = namedtuple("Nodes", "domestic, wild, captive")
    nodes = np.array([tree_seq.individual(i).nodes for i in pyslim.individuals_alive_at(tree_seq, 0)])
    pop = np.array([tree_seq.individual(i).population for i in pyslim.individuals_alive_at(tree_seq, 0)])
    node_tuple = Nodes(domestic=nodes[pop == 0], scot=nodes[pop == 1], captive=nodes[pop == 2])

    return node_tuple


def add_seed_suffix_to_file(filename, seed):
    """Helper function adds _{random_seed} before the last dot in filenames. Avoids clashes in filenames."""
    dot_index = filename.rfind('.')
    filename = "{}_{}{}".format(filename[:dot_index], seed, filename[dot_index:])
    return filename


def tree_summary(tree_seq):
    """Prints summary of a tree sequence"""
    tree_heights = []
    for tree in tree_seq.trees():
        for root in tree.roots:
            tree_heights.append(tree.time(root))
    print("Number of trees: {}".format(tree_seq.num_trees))
    print("Trees coalesced: {}".format(sum([t.num_roots == 1 for t in tree_seq.trees()])))
    print("Tree heights: max={}, min={}, median={}".format(max(tree_heights), min(tree_heights),
                                                           np.median(tree_heights)))
    print("Number of alive individuals: {}".format(len(pyslim.individuals_alive_at(tree_seq, 0))))
    print("Number of samples: {}".format(tree_seq.num_samples))
    print("Number of populations: {}".format(tree_seq.num_populations))
    print("Number of variants: {}".format(tree_seq.num_mutations))
    print("Sequence length: {}".format(tree_seq.sequence_length))


class GenotypeData:
    """
        Collates results from the simulation or real data in a format ideal for scikit-allel analysis.
        Genotype values greater than one when using a callset are set to 1 (i.e. assuming biallelic).
        :param tree_seq: tskit tree sequence, to generate results from simulations
        :param callset: scikit allel callset dictionary (from allel.read_vcf) for using real data
        :param subpops: dictionary of subpop names and corresponding indexes in genotypes when using callset
        :param seq_length: length of sequence when using callset
        :return: GenotypeData object with attributes calculated
    """
    def __init__(self, tree_seq=None, callset=None, subpops=None, seq_length=None):
        self.genotypes = None
        self.positions = None
        self.allele_counts = None
        self.subpops = subpops
        self.seq_length = seq_length

        if tree_seq is None and callset is None:
            raise ValueError("Either tree_seq or callset must be specified")

        if tree_seq is not None and callset is not None:
            raise ValueError("Only one of tree_seq or callset should be specified")

        if tree_seq is not None:
            self._initialize_from_tree_seq(tree_seq)

        if callset is not None:
            self._initialize_from_callset(callset)

    def _initialize_from_tree_seq(self, tree_seq):
        if self.subpops is not None:
            logging.warning("Ignoring subpops parameter, using info in tree_seq")
        if self.seq_length is not None:
            logging.warning("Ignoring seq_length parameter, using info in tree_seq")

