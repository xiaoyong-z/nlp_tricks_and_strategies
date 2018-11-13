import pandas as pd
data = pd.read_csv("  ")
def filter_map(arr):
    res = ""
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0'and c != '\n' and c != '\ufeff' and c != '\r':
            res += c
    return res
data.content = data.content.map(lambda x: filter_map(x))
