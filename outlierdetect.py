#!/usr/bin/env python
# encoding: utf-8
"""
outlierdetect.py

Created by Ben Birnbaum on 2012-08-27.
benjamin.birnbaum@gmail.com
"""

import collections
import itertools
import math
from matplotlib import pyplot as plt  # TODO: do I really want this dependency?
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


_SCORE_COLORS = [  # From http://colorbrewer2.org/
    (1.0, 1.0, 1.0),
    (0.945, 0.933, 0.964),
    (0.843, 0.709, 0.847),
    (0.874, 0.396, 0.690),
    (0.866, 0.109, 0.466),
    (0.596, 0.0, .262745),
]


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

def _normalize_counts(counts, val=1):
    """Helper function to normalize a dictionary of counts.

    It normalizes the counts to add up to val.
    counts should be a dictionary of the form {val1 : count1, val2 : count2, ...}.
    Returns a dictionary of the same form.
    """
    n = sum([counts[k] for k in counts.keys()])
    frequencies = {}
    for r in counts.keys():
        frequencies[r] = val * float(counts[r]) / float(n)
    return frequencies


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
    """TODO: comment."""
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


def _write_or_show_plot(filename):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        print "Wrote file " + filename


def _compute_color_number(value, max_value, cutoffs=None):
    num_colors = len(_SCORE_COLORS)
    if cutoffs is None or len(cutoffs) != num_colors - 1:
        norm_score = value / max_value
        return int(math.floor(norm_score * (num_colors - 1)))
    else:
        color_number = 0
        for i in range(num_colors - 1):
            if value > cutoffs[i]:
                color_number = i + 1
        return color_number


########################################## Public functions ########################################

if _STATS_AVAILABLE:
    def run_mma(data, aggregation_column, categorical_columns):
        """TODO: comment."""
        return _run_alg(data, aggregation_column, categorical_columns, MultinomialModel())


def run_sva(data, aggregation_column, categorical_columns):
    """TODO: comment."""
    return _run_alg(data, aggregation_column, categorical_columns, SValueModel())


def plot_scores(scores, leftpad=1.5, rightpad=1.9, toppad=1.5, bottompad=0.1, scale=1, filename=None, cutoffs=None):
    """Draws a 2-D heat map of outlier scores.
    
    Arguments:
    scores -- dict of aggregation_unit -> column -> score
    leftpad -- inches to add left of the heat map
    rightpad -- inches to add right of the heat map
    toppad -- inches to add above the heat map
    bottompad -- inches to add below the heat map
    scale -- scaling factor to apply after padding figured.  Affects everything but font.
    filename -- if specified, gives the file name to which the plot will be saved.
        If not specified, the plot is shown using the pylab.show() function.
    cutoffs -- s-value cutoffs for different colors in heatmaps.  If none or a list of wrong
        size, the cutoffs will be chosen automatically.
    """
    plot_scores_list([scores], [''], num_cols=1,
        leftpad=leftpad, rightpad=rightpad, toppad=toppad, bottompad=bottompad,
        scale=scale, filename=filename, cutoffs=cutoffs)


def plot_scores_list(scores_list, titles_list, num_cols=1,
        leftpad=1.5, rightpad=1.75, toppad=1.5, bottompad=0.3, scale=1, filename=None, cutoffs=None):
    """Draws a set of 2-D heat maps of a list of outlier scores, all on the same scale.
    
    Arguments:
    scores_list -- a list of dicts of aggregation_unit -> column -> score
    titles_list -- a list of titles for each set of outlier scores
    num_cols -- the number of columns on which to display the heat maps
    leftpad -- inches to add to the left of each heat map
    rightpad -- inches to add to the right of each heat map
    toppad -- inches to add above above each heat map
    bottompad -- inches to add below each heat map
    scale -- scaling factor to apply after padding figured.  Affects everything but font.
    filename -- if specified, gives the file name to which the plot will be saved.
        If not specified, the plot is shown using the pylab.show() function.
    cutoffs -- s-value cutoffs for different colors in heatmaps.  If none or a list of wrong
        size, the cutoffs will be chosen automatically.

    Raises:
    ValueError if the length of scores_list and titles_list is not equal.
    """
    if len(scores_list) != len(titles_list):
        raise ValueError("Length of scores_list must equal length of titles_list")

    # The relative values of these constants is the only thing that matters.
    SEP = 10  # row height and column width, in abstract axis units
    RAD = 4   # radius of circles, in abstract axis units
    UNITS_IN_INCH = 25.0  # Number of abstract axis units per inch
    
    # Compute useful variables and create figure.
    num_scores = len(scores_list)
    agg_units = sorted(scores_list[0].keys())
    cols = sorted(scores_list[0][agg_units[0]].keys())
    m, n = len(cols), len(agg_units)
    xmax, ymax = m * SEP, n * SEP
    max_score = max([scores[agg_unit][col]
        for scores in scores_list
        for agg_unit in agg_units
        for col in cols])
    num_colors = len(_SCORE_COLORS)
    num_rows = num_scores / num_cols if num_scores % num_cols == 0 else num_scores / num_cols + 1
    figlength = num_cols * ((m * SEP) / UNITS_IN_INCH + leftpad + rightpad)
    figheight = num_rows * ((n * SEP) / UNITS_IN_INCH + toppad + bottompad)
    wspace = num_cols * (leftpad + rightpad)  # Total amount of horizontal space
    hspace = num_rows * (toppad + bottompad)  # Total amount of vertical space
    plotlength = (figlength - wspace) / num_cols  # Length of one plot
    plotheight = (figheight - hspace) / num_rows  # Height of one plot
    
    fig = plt.figure(figsize=(figlength * scale, figheight * scale))    
    
    # Iterate through scores to create subplots.
    for i in range(len(scores_list)):
        scores = scores_list[i]
        title = titles_list[i]
        
        # Setup basic plot and ticks.
        plt.subplot(num_rows, num_cols, i + 1)
        plt.xlim((0, xmax))
        plt.ylim((0, ymax))
        plt.gca().xaxis.set_ticks_position('top')
        plt.xticks([SEP / 2 + x for x in range(0, xmax, SEP)], cols, rotation=90)
        plt.yticks([SEP / 2 + x for x in range(0, ymax, SEP)], agg_units)
        plt.xlabel(title)
    
        # Draw the circles.
        for i in range(m):
            for j in range(n):
                score = scores[agg_units[j]][cols[i]]
                color_number = _compute_color_number(score, max_score, cutoffs)
                color = _SCORE_COLORS[color_number]
                cir = plt.Circle(((i + 0.5) * SEP, (j + 0.5) * SEP), RAD, fc=color, edgecolor='None')
                plt.gca().add_patch(cir)
        
        # Create legend using dummy patches having the appropriate face color.
        
        if cutoffs is None:    
            shown_cutoffs = []
        else:
            shown_cutoffs = ([0] + cutoffs)[::-1]
        patches = []
        for i in range(num_colors)[::-1]:
            # The x-y coordinates of the circles don't matter since we're not actually
            # adding them to the plot.
            patches.append(plt.Circle((0, 0), fc=_SCORE_COLORS[i], edgecolor='None'))
            if cutoffs is None:
                shown_cutoffs.append("%.2f" % (i * (max_score / num_colors)))
        # The values 0.7 and -0.01 are just what seemed to work best.
        # The weirdness necessary to place the legend seems to be partly because of
        # the weirdness of subplots_adjust().  Maybe there is a more precise way to place
        # the plots precisely, using figure coordinates that would fix this.
        plt.legend(patches, shown_cutoffs, loc='lower right', title='s-value\ncutoffs',
            bbox_to_anchor=((plotlength + 0.7 * rightpad) / plotlength, -0.01))

    # Fix white between plots (using fractional figure coordinates).
    fig.subplots_adjust(
        left=(leftpad / figlength),
        right=(1 - rightpad / figlength),
        bottom=(bottompad / figheight),
        top=(1 - toppad / figheight),
        wspace=(wspace / figlength),
        hspace=(hspace / figheight))    
    _write_or_show_plot(filename)
