#!/usr/bin/env python
# encoding: utf-8
"""
outlierdetect.py

Created by Ben Birnbaum on 2012-08-27.
benjamin.birnbaum@gmail.com


This module provides functions that implement the Multinomial Model Algorithm (MMA) and the s-Value
Algorithm (SVA), as described in

B. Birnbaum, B. DeRenzi, A. D. Flaxman, and N. Lesh.  Automated quality control for mobile data
collection. In DEV ’12, pages 1:1–1:10, 2012.

B. Birnbaum. Algorithmic approaches to detecting interviewer fabrication in surveys.  Ph.D.
Dissertation, Univeristy of Washington, Department of Computer Science and Engineering, 2012.

(See http://homes.cs.washington.edu/~birnbaum/pubs.html for PDF versions of these papers.)

This module is designed to work for two Python data structures: numpy.recarray and
pandas.DataFrame.  Both of these data structures consist of rows of structured data, where
columns can be accessed by string identifiers.  One of these columns must be a special column,
which is called the "aggreagation unit" column.  Each entry in this column is an identifier
which an outlier score can be computed.  For example, if the data is from a survey, the aggregation
column might be the column that lists which interviewer performed the survey; then we would be
interested in obtaining outlier scores for the interviewers.  In other situations, the aggregation
column might be different.  It just depends on what you want to compute outlier scores for.

Note that the MMA and SVA algorithms work only for categorical data.  You must specify which
categorical columns you want to compute outlier scores for.

This module requires numpy, and the implementation of MMA requires scipy.  (The module will load
without scipy, but MMA will not be available.)

The algorithms should be called by the public methods run_mma() and run_sva().


Examples:

# With pandas.DataFrame:
import pandas as pd
import outlierdetect
data = pd.read_csv('survey_data.csv')
sva_scores = outlierdetect.run_sva(data, 'interviewer_id', ['available', 'cough', 'fever'])

# With numpy.recarray:
from matplotlib import mlab
import outlierdetect
data = mlab.csv2rec('survey_data.csv')
sva_scores = outlierdetect.run_sva(data, 'interviewer_id', ['available', 'cough', 'fever'])
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

_FLOAT_EQ_DELTA = 0.000001  # For comparing float equality

############################################## Models ##############################################
#
# Models define the core logic for computing an outlier score for a given algorithm.  Each model
# must implement a compute_outlier_scores() method defining this logic.


if _STATS_AVAILABLE:
    class MultinomialModel:
        """Model implementing MMA.  Requries scipy module."""


        def compute_outlier_scores(self, frequencies):
            """Computes the SVA outlier scores fo the given frequencies dictionary.
        
            Args:
                frequencies: dictionary of dictionaries, mapping (aggregation unit) -> (value) ->
                    (number of times aggregation unit reported value).
            
            Returns:
                dictionary mapping (aggregation unit) -> (MMA outlier score for aggregation unit).
            """
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
            
            Returns:
                the X^2 statistic for the actual frequencies, given the expected frequencies.
            """
            rng = expected.keys()
            if actual.keys() != rng:
                raise Exception("Ranges of two frequencies are not equal.")
            num_observations = sum([actual[r] for r in rng])
            if abs(num_observations - sum([expected[r] for r in rng])) > _FLOAT_EQ_DELTA:
                raise Exception("Frequencies must sum to the same value.")
            return sum([(actual[r] - expected[r])**2 / max(float(expected[r]), 1.0)
                for r in expected.keys()])


        def _sum_frequencies(self, agg_unit, frequencies):
            """Sums frequencies for each aggregation unit except the given one.
            
            Args:
                agg_unit: the aggregation unit of concern.
                frequencies: dictionary of dictionaries, mapping (aggregation unit) -> (value) ->
                    (number of times aggregation unit reported value).
            
            Returns:
                a dictionary mapping (value) -> (number of times all aggregation units apart from
                agg_unit reported this value)

            """
            # Get the range from the frequencies dictionary.  Assumes that the range is the same
            # for each aggregation unit in this distribution.  Bad things may happen if this is not
            # the case.
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
    """Model implementing SVA."""


    def compute_outlier_scores(self, frequencies):
        """Computes the SVA outlier scores fo the given frequencies dictionary.
        
        Args:
            frequencies: dictionary of dictionaries, mapping (aggregation unit) -> (value) ->
                (number of times aggregation unit reported value).
            
        Returns:
            dictionary mapping (aggregation unit) -> (SVA outlier score for aggregation unit).
        """
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
        
        Args:
            value_dict: dictionary of the form (aggregation unit) -> (value).
        Returns:
            dictionary of the same form as value_dict, where the values are normalized as described
            above.
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
    """Normalizes a dictionary of counts, such as those returned by _get_frequencies().

    Args:
        counts: a dictionary mapping value -> count.
        val: the number the counts should add up to.
    
    Returns:
        dictionary of the same form as counts, except where the counts have been normalized to sum
        to val.
    """
    n = sum([counts[k] for k in counts.keys()])
    frequencies = {}
    for r in counts.keys():
        frequencies[r] = val * float(counts[r]) / float(n)
    return frequencies


def _get_frequencies(data, col, col_vals, agg_col, agg_unit):
    """Computes a frequencies dictionary for a given column and aggregation unit.
    
    Args:
        data: numpy.recarray or pandas.DataFrame containing the data.
        col: name of column to compute frequencies for.
        col_vals: a list giving the range of possible values in the column.
        agg_col: string giving the name of the aggregation unit column for the data.
        agg_unit: string giving the aggregation unit to compute frequencies for.

    Returns:
        A dictionary that maps (column value) -> (number of times agg_unit has column value in
        data).
    """
    frequencies = {}
    for col_val in col_vals:
        frequencies[col_val] = 0
        # We can't just use collections.Counter() because frequencies.keys() is used to determine
        # the range of possible values in other functions.
    if _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        grouped = data[data[agg_col] == agg_unit].groupby(col)
        for name, group in grouped:
            frequencies[name] = len(group)
    else:  # Assumes it is an np.ndarray
        for row in itertools.ifilter(lambda row : row[agg_col] == agg_unit, data):
            frequencies[row[col]] += 1
    return frequencies


def _run_alg(data, agg_col, cat_cols, model):
    """Runs an outlier detection algorithm, taking the model to use as input.
    
    Args:
        data: numpy.recarray or pandas.DataFrame containing the data.
        agg_col: string giving the name of aggregation unit column.
        cat_cols: list of the categorical column names for which outlier values should be computed.
        model: object implementing a compute_outlier_scores() method as described in the comments
            in the models section.
    
    Returns:
        A dictionary of dictionaries, mapping (aggregation unit) -> (column name) ->
        (outlier score).
    """
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
