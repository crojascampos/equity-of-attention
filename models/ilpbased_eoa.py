import math
import numpy

from mip import Model, minimize, xsum, MINIMIZE, CBC, BINARY
from helpers.extra_functions import normalize_array
from typing import Union

class ILPBasedEOA:
    def __init__(self, ranking: Union[list, tuple, numpy.ndarray], min_val: Union[float, list, tuple, numpy.ndarray], max_val: Union[float, list, tuple, numpy.ndarray]) -> None:
        """
        Initialization of ILPBasedEOA

        Parameters
        ----------
        ranking : 1 or 2 dimension list, tuple or numpy array
            Relevance scores for 1 or multiple subjects.
        min_val : float or array of floats
            Minimum value (or values) of the ranking (or rankings)
        max_val : float or array of floats
            Maximum value (or values) of the ranking (or rankings)
        """
        # Converts the ranking to a numpy array if its not
        if type(ranking) != numpy.ndarray:
            og_ranking = numpy.array(ranking)
        else:
            og_ranking = ranking
        # Converts the ranking to a two dimensional list if it's a one dimensional array-like
        if og_ranking.ndim == 1:
            og_ranking = numpy.array([og_ranking])
        # Raises exception if the ranking is not a 2 dimensional array
        if og_ranking.ndim != 2:
            raise Exception(
                "'ranking' must be a 1 or 2 dimension list, tuple or numpy array")

        # Saves the min and max values as a list of values
        self.min_val = [min_val] if type(min_val) == int or type(min_val) == float else min_val
        self.max_val = [max_val] if type(max_val) == int or type(max_val) == float else max_val

        # Checks if the lengths of the minimum and maximum values match the length of the rankings
        if len(self.min_val) != len(og_ranking) or len(self.max_val) != len(og_ranking):
            raise Exception(
                "'min_val' and 'max_val' must have the same length as the rankings")

        # Normalize (between 0 and 1) and saves the rankings as a attribute
        self.ranking = [normalize_array(arr_r, self.min_val[r], self.max_val[r]) for r, arr_r in enumerate(og_ranking)]

        # Attributes for the quantity of subjects and rankings
        self.qty_subjects = len(self.ranking[0])
        self.qty_rankings = len(self.ranking)

        # Attributes assigned when preparing the model
        self.k = self.pref_n = self.prob = self.theta = 0

        # Iteration attributes
        self.it_ranking = 0
        self.it_prefilter = 0

        # Base attributes
        self.accummA = self.accummR = 0
        self.idcg_constant = 0

        # MIP attributes
        self.mip_model = 0
        self.mip_vars = 0
        self.mip_prepared = False

        # Functions for getting the position bias, unfairness value and ranking quality
        self.pos_bias = lambda pos: 0 if pos > self.k else self.prob * math.pow(1-self.prob, pos-1)
        self.unfairness = lambda i, j: 0 if self.mip_vars[i][j] == 0 else math.fabs(self.accummA[i] + self.pos_bias(j) - (self.accummR[i] + self.it_prefilter[self.this_ranking][i]))
        self.ranking_quality = lambda i, j: 0 if self.mip_vars[i][j] == 0 else (math.pow(2, self.it_prefilter[self.this_ranking][i])-1) / math.log2(j+2)

        # Counters
        self.this_ranking = 0
        self.this_iteration = 0

    def prepare(self, prob: float, k: int, pref_n: int, theta: float) -> None:
        """
        Prepares the model

        Arguments
        ---------

        prob : float
            Probability that any subject will be chosen.
        k : int
            Amount of top subjects to consider in calculations.
        pref_n : int
            Amount of subjects to prefilter on each iteration.
        theta : float
            Multiplicative threshold for IDCG@k.
        """

        # Sets the flag indicating that the model is not prepared
        self.mip_prepared = False

        # Checks that the probability and theta are between 0 (inclusive) and 1 (inclusive)
        if prob < 0 or 1 < prob:
            raise Exception(
                "Could not prepare -> 'prob' must be within 0 (inclusive) and 1 (inclusive)")
        if theta < 0 or 1 < theta:
            raise Exception(
                "Could not prepare -> 'theta' must be within 0 (inclusive) and 1 (inclusive)")

        # Sets the parameters to the corresponding attributes
        self.k, self.pref_n = min(k, pref_n), pref_n
        self.prob, self.theta = prob, theta

        # Sets the counters to the start
        self.this_ranking = 0
        self.this_iteration = 0

        # Gets the ordered ranking positions without modifying the original rankings
        self.it_ranking = [numpy.argsort(-r) for r in self.ranking]
        # Defines the prefilter array (NOT YET READY)
        self.it_prefilter = 0

        # Defines the initial accummulated attributes as 0
        self.accummA = self.accummR = [0] * self.qty_subjects
        # Calculates the IDCG@k constant for each ranking
        self.idcg_constant = [numpy.sum([(math.pow(2, rel_i)-1) / math.log2(i+2) for i, rel_i in enumerate(arr_r[self.it_ranking[r]])]) for r, arr_r in enumerate(self.ranking)]

        # Creates MIP model
        self.mip_model = Model(sense=MINIMIZE, solver_name=CBC)
        # Creates MIP variables
        self.mip_vars = [[self.mip_model.add_var("vars({},{})".format(
            i, j), var_type=BINARY) for j in range(self.pref_n)] for i in range(self.pref_n)]
        # Sets Objective function
        self.mip_model.objective = minimize(xsum(self.unfairness(i, j)*self.mip_vars[i][j] for i in range(self.pref_n) for j in range(self.pref_n)))
        # Sets Constraint: makes sure that this ranking dcg is the same or better than the original dcg, within a threshold theta
        self.mip_model += xsum(self.ranking_quality(i, j)*self.mip_vars[i][j] for j in range(self.k) for i in range(self.pref_n)) >= self.theta * self.idcg_constant[self.this_ranking]
        # Sets Constraint: makes sure that the solution is a bijective mapping of subjects
        for j in range(self.pref_n):
            self.mip_model += xsum(self.mip_vars[i][j] for i in range(self.pref_n)) == 1
        for i in range(self.pref_n):
            self.mip_model += xsum(self.mip_vars[i][j] for j in range(self.pref_n)) == 1

        # Set flag indicating that the MIP is prepared
        self.mip_prepared = True

    def start(self, it:int) -> None:
        """
        Starts the model

        Arguments
        ---------
        it : integer
            Number of iterations to calculate
        """

        if self.mip_prepared != True:
            raise Exception(
                "The model is not prepared, call the prepare method first before starting the model.")

        print("Prefiltering...")
        self.it_prefilter = []
        for i, val in enumerate(self.it_ranking):
            print(val[:self.k])
            temp = numpy.concatenate(
                (val[:self.k], numpy.argsort(
                    [self.accummA[i] - (self.accummR[i] + rel) for rel in self.ranking[i][:self.k]])[:self.pref_n]))
            print(temp)
            self.it_prefilter = numpy.concatenate((self.it_prefilter, temp))
            print(self.it_prefilter)

        for iteration in range(it):
            self.this_iteration = iteration
            for r in range(self.qty_rankings):
                self.this_ranking = r
                print("Optimizing... iteration {}, ranking {}".format(iteration, r))
                self.mip_model.optimize()
                # TODO: Save the order generated as this ranking new order for the next iteration
            # TODO: Implement iteration
