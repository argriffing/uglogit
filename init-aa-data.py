"""
Convert a file into binary outputs.

The first column is assumed to be the response label.
The remaining columns are assumed to be predictors.
An intercept is added to the predictor set.

"""

import itertools
import sys
import struct
import csv


# Copypasted directly from official itertools recipes.
def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def main():

    # read the csv file from stdin
    endog = []
    exog = []
    reader = csv.reader(sys.stdin, delimiter='\t')
    for row in reader:
        endog_element = row[0]
        exog_row = [float(x) for x in row[1:]] + [1.0]
        endog.append(endog_element)
        exog.append(exog_row)

    # Map labels from strings to integers.
    index_to_label = list(unique_everseen(endog))
    label_to_index = dict((x, i) for i, x in enumerate(index_to_label))

    # Write the binary file that has the prediction data.
    print 'processing prediction data...'
    with open('aa.predict', 'wb') as fout_predict:
        fmt = 'd' * len(exog[0])
        for i, row in enumerate(exog):
            if not (i+1) % 100000:
                print 'row', i+1, 'of', len(exog)
            fout_predict.write(struct.pack(fmt, *row))

    # Write the binary file that has the response data.
    print 'processing response data...'
    with open('aa.respond', 'wb') as fout_respond:
        fmt = 'i'
        for i, label in enumerate(endog):
            if not (i+1) % 100000:
                print 'row', i+1, 'of', len(exog)
            idx = label_to_index[label]
            fout_respond.write(struct.pack(fmt, idx))

    # Report a few numbers.
    print len(endog), 'observations'
    print len(exog[0]), 'predictors including the intercept'
    print len(label_to_index), 'categories including the reference category'


if __name__ == '__main__':
    main()

