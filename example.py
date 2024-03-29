#!/usr/bin/env python
# encoding: utf-8
"""
example.py

Created by Ben Birnbaum on 2012-12-02.
benjamin.birnbaum@gmail.com

Example use of outlierdetect.py module.
"""

from __future__ import print_function
from matplotlib import mlab
import outlierdetect
import pandas as pd


DATA_FILE = 'example_data.csv'


def print_scores(scores):
    for interviewer in scores.keys():

        print("%s" % interviewer)

        for column in scores[interviewer].keys():
            
            score = scores[interviewer][column]['score']
            observed_frequencies = scores[interviewer][column]['observed_freq']
            expected_frequencies = scores[interviewer][column]['expected_freq']
            p_value = scores[interviewer][column]['p_value']

            print("Observed Frequencies: %s" % observed_frequencies)
            print("Expected Frequencies: %s" % expected_frequencies)
            print("P-Value: %d" % p_value)

if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE)  # Uncomment to load as pandas.DataFrame.
    # data = mlab.csv2rec(DATA_FILE)  # Uncomment to load as numpy.recarray.

    # Compute SVA outlier scores.
    (sva_scores, _) = outlierdetect.run_sva(data, 'interviewer_id', ['cough', 'fever'])
    print("SVA outlier scores")
    print_scores(sva_scores)

    # Compute MMA outlier scores.  Will work only if scipy is installed.
    if hasattr(outlierdetect, 'run_mma'):
        (mma_scores, _) = outlierdetect.run_mma(data, 'interviewer_id', ['cough', 'fever'])
        print("\nMMA outlier scores")
        print_scores(mma_scores)
