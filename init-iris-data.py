"""
Write some binary files.
iris.predict
iris.respond
"""

import itertools
import struct

import statsmodels
import statsmodels.api as st

IRIS_PREDICTOR_COUNT = 4


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
    """
    Use the famous iris data set.
    """

    # Read some data.
    iris = statsmodels.datasets.get_rdataset('iris', 'datasets')
    y = iris.data.Species
    print y.values
    x = iris.data.ix[:, :IRIS_PREDICTOR_COUNT]
    x = st.add_constant(x, prepend=False)
    print x.values

    # Map labels from strings to integers.
    index_to_label = list(unique_everseen(y.values))
    label_to_index = dict((x, i) for i, x in enumerate(index_to_label))

    # Write the binary file that has the prediction data.
    with open('iris.predict', 'wb') as fout_respond:
        fmt = 'd'
        for row in x.values:
            for v in row:
                fout_respond.write(struct.pack(fmt, float(v)))

    # Write the binary file that has the response data.
    with open('iris.respond', 'wb') as fout_predict:
        fmt = 'i'
        for label in y.values:
            idx = label_to_index[label]
            fout_predict.write(struct.pack(fmt, idx))

    # Report a few numbers.
    print len(y.values), 'observations'
    print x.values.shape[1], 'predictors including the intercept'
    print len(label_to_index), 'categories including the reference category'


if __name__ == '__main__':
    main()

