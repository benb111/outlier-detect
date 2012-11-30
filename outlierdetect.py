#!/usr/bin/env python
# encoding: utf-8
"""
outlierdetect.py

Created by Ben Birnbaum on 2012-08-27.
benjamin.birnbaum@gmail.com
"""

import collections
import itertools
import numpy as np
import sys

# Import optional dependencies
_PANDAS_AVAILABLE = False
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pass
_STATS_AVAILABLE = False
try:
    from scipy import stats
    _STATS_AVAILABLE = True
except ImportError:
    sys.stderr.write('Cannot import scipy.  Some models may not be available.\n')
    sys.stderr.flush()
    pass


############################################## Models ##############################################
#
# Each model must define a method with the signature compute_outlier_scores(frequencies).  The
# parameter frequencies is a dictionary containing the distribution of response values for each
# aggregation unit.  The method must return a dictionary giving the outlier score for each
# aggregation unit.
#
# For example, the frequencies parameter might look like
#
# frequencies = {
#     'interviewer1' : {
#         'yes' : 23,
#         'no'  : 12,
#         'NA'  : 3
#     },
#     'interviewer2' : {
#         'yes' : 8,
#         'no'  : 25,
#         'NA'  : 3
#     }
# }
#
# and the method might return something like
# 
# {
#     'interviewer1' : 7.34,
#     'interviewer2' : 2.30
# }


if _STATS_AVAILABLE:
    class MultinomialModel:
        """Model to compute outlier scores based on the chi^2 test for multinomial data.
    
        See B. Birnbaum, B. DeRenzi, A. D. Flaxman, and N. Lesh. Automated quality control for mobile
        data collection. In DEV ’12, pages 1:1–1:10, 2012.
    
        Requries scipy module.
        """


        def compute_outlier_scores(self, frequencies):
            if len(frequencies.keys()) < 2:
                raise Exception("There must be at least 2 aggregation units.")
            rng = frequencies[frequencies.keys()[0]].keys()
            outlier_scores = {}
            for agg_unit in frequencies.keys():
                expected_counts = _normalize_counts(
                    self._sum_frequencies(agg_unit, frequencies),
                    val=sum([frequencies[agg_unit][r] for r in rng]))
                x2 = self._compute_x2_statistic(expected_counts, frequencies[agg_unit])
                # logsf gives the log of the survival function (1 - cdf).
                outlier_scores[agg_unit] = -stats.chi2.logsf(x2, len(rng) - 1)
            return outlier_scores


        def _compute_x2_statistic(self, expected, actual):
            """Computes the X^2 statistic for observed frequencies.

            Args:
                expected: a dictionary giving the expected frequencies, e.g.,
                    {'y' : 13.2, 'n' : 17.2}
                actual: a dictionary in the same format as the expected dictionary
                    (with the same range) giving the actual distribution.
            """
            rng = expected.keys()
            if actual.keys() != rng:
                raise Exception("Ranges of two frequencies are not equal.")
            num_observations = sum([actual[r] for r in rng])
            if abs(num_observations - sum([expected[r] for r in rng])) > 0.0001:
                raise Exception("Frequencies must sum to the same value.")
            return sum([(actual[r] - expected[r])**2 / max(float(expected[r]), 1.0)
                for r in expected.keys()])


        def _sum_frequencies(self, agg_unit, frequencies):
            """Sums frequencies for each aggregation unit except the given one."""
            # Get the range from the frequencies dictionary.  Assumes that the
            # range is the same for each aggregation unit in this distribution.
            rng = frequencies[agg_unit].keys()
            all_frequencies = {}
            for r in rng:
                all_frequencies[r] = 0
            for other_agg_unit in frequencies.keys():
                if other_agg_unit == agg_unit:
                    continue
                for r in rng:
                    all_frequencies[r] += frequencies[other_agg_unit][r]        
            return all_frequencies


class SValueModel:
    """Computes s-value outlier scores.
    
    See B. Birnbaum, B. DeRenzi, A. D. Flaxman, and N. Lesh. Automated quality control for mobile
    data collection. In DEV ’12, pages 1:1–1:10, 2012.
    """


    def compute_outlier_scores(self, frequencies):
        if (len(frequencies.keys()) < 2):
            raise Exception("There must be at least 2 aggregation units.")
        rng = frequencies[frequencies.keys()[0]].keys()
        normalized_frequencies = {}
        for j in frequencies.keys():
            normalized_frequencies[j] = _normalize_counts(frequencies[j])
        medians = {}    
        for r in rng:
            medians[r] = np.median([normalized_frequencies[j][r]
                for j in normalized_frequencies.keys()])
        outlier_values = {}
        for j in frequencies.keys():
            outlier_values[j] = 0
            for r in rng:
                outlier_values[j] += abs(normalized_frequencies[j][r] - medians[r])
        return self._normalize(outlier_values)
    
    
    def _normalize(self, value_dict):
        """Divides everything in value_dict by the median of values.

        If the median is less than 1 / (# of aggregation units), it divides everything by
        (# of aggregation units) instead.
        """
        median = np.median([value_dict[i] for i in value_dict.keys()])
        n = len(value_dict.keys())
        if median < 1.0 / float(n):
            divisor = 1.0 / float(n)
        else:
            divisor = median
        return_dict = {}
        for i in value_dict.keys():
            return_dict[i] = float(value_dict[i]) / float(divisor)
        return return_dict


########################################## Helper functions ########################################

def _normalize_counts(counts, val=1):
    """Normalizes a dictionary of counts.

    It normalizes the counts to add up to val.
    counts should be a dictionary of the form {val1 : count1, val2 : count2, ...}.
    Returns a dictionary of the same form.
    """
    n = sum([counts[k] for k in counts.keys()])
    frequencies = {}
    for r in counts.keys():
        frequencies[r] = val * float(counts[r]) / float(n)
    return frequencies


def _get_frequencies(data, col, col_vals, agg_col, agg_unit):
    """TODO: comment."""
    frequencies = {}
    for col_val in col_vals:
        frequencies[col_val] = 0
        # (We can't just use collections.Counter() because frequencies.keys() is used to determine
        # the range of possible values in other functions.)
    if _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        grouped = data[data[agg_col] == agg_unit].groupby(col)
        for name, group in grouped:
            frequencies[name] = len(group)
    else:  # Assumes it is an np.ndarray
        for row in itertools.ifilter(lambda row : row[agg_col] == agg_unit, data):
            frequencies[row[col]] += 1
    return frequencies


def _run_alg(data, agg_col, cat_cols, model):
    """Runs an outlier detection algorithm, taking the model to use as input."""
    agg_units = sorted(np.unique(data[agg_col]))
    outlier_scores = collections.defaultdict(dict)
    for col in cat_cols:
        col_vals = sorted(np.unique(data[col]))
        frequencies = {}
        for agg_unit in agg_units:
            frequencies[agg_unit] = _get_frequencies(data, col, col_vals, agg_col, agg_unit)
        outlier_scores_for_col = model.compute_outlier_scores(frequencies)
        for agg_unit in agg_units:
            outlier_scores[agg_unit][col] = outlier_scores_for_col[agg_unit]
    return outlier_scores


########################################## Public functions ########################################

if _STATS_AVAILABLE:
    def run_mma(data, aggregation_column, categorical_columns):
        """Runs the MMA algorithm (requires scipy module).
        
        Args:
            data: numpy.recarray or pandas.DataFrame containing the data.
            aggregation_column: a string giving the name of aggregation unit column.
            categorical_columns: a list of the categorical column names for which outlier values
                should be computed.
        
        Returns:
            A dictionary of dictionaries, mapping (aggregation unit) -> (column name) ->
            (mma outlier score).
        """
        return _run_alg(data, aggregation_column, categorical_columns, MultinomialModel())


def run_sva(data, aggregation_column, categorical_columns):
        """Runs the SVA algorithm.
        
        Args:
            data: numpy.recarray or pandas.DataFrame containing the data.
            aggregation_column: a string giving the name of aggregation unit column.
            categorical_columns: a list of the categorical column names for which outlier values
                should be computed.
        
        Returns:
            A dictionary of dictionaries, mapping (aggregation unit) -> (column name) ->
            (sva outlier score).
        """
    return _run_alg(data, aggregation_column, categorical_columns, SValueModel())
