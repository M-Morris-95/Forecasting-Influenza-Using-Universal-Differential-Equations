import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr
import copy
import time

def return_folds(x, min_train_size = 2000, validation_size = 365, num_folds=5):
    length_train = x.shape[0]-min_train_size-validation_size
    val_starts = np.linspace(0, length_train, num_folds).astype(int) + min_train_size

    folds = []
    for val_start in val_starts:
        tr = np.linspace(0, val_start-1, val_start).astype(int)
        val=np.linspace(val_start, val_start+validation_size-1, validation_size).astype(int)
        folds.append({'train':tr, 'val':val})
    return folds

class DataConstructor():
    def __init__(self, test_season, data_season, gamma, window_size, n_queries=100, selection_method='distance',
                 selection_similarity_threshold=2.5, selection_correlation_threshold=0.9, root='Data/'):
        self.test_season = test_season
        self.gamma = gamma
        self.window_size = window_size
        self.root = root
        self.data_season = data_season
        self.n_queries = n_queries
        self.selection_similarity_threshold = selection_similarity_threshold
        self.selection_correlation_threshold = selection_correlation_threshold
        self.selection_method = selection_method
        return

    def __call__(self):
        self.get_ili()
        self.get_queries()
        self.get_dates()
        self.query_selection()


    def build(self):
        wILI = copy.deepcopy(self.daily_wILI)
        queries = copy.deepcopy(self.queries[self.selected_queries])

        queries.index = queries.index-dt.timedelta(days=14)
        queries['wILI'] = wILI

        # queries = queries[np.invert(queries.isna().any(1))]
        queries = queries[np.invert(np.any(queries.isna().values, 1))]

        dates = pd.date_range(queries.index[self.window_size-1], queries.index[-(self.gamma+1)])
        inputs = []
        outputs = []
        for date in dates:
            temp = copy.deepcopy(queries.loc[date-dt.timedelta(days=self.window_size-1):date+dt.timedelta(days=14)])
            temp.loc[date:date+dt.timedelta(days=14), 'wILI'] = 0

            inputs.append(temp)
            outputs.append(queries.loc[date+dt.timedelta(days=1):date+dt.timedelta(days=self.gamma),'wILI'])
        inputs = np.asarray(inputs)
        outputs= np.asarray(outputs)

        dates = np.asarray(list(dates))
        return inputs, outputs, dates, self.test_dates

    def get_ili(self):
        self.wILI = \
            pd.read_csv(os.path.join(self.root, 'ILI_rates', 'national_flu.csv'), index_col=-1, parse_dates=True)[
                'weighted_ili']

        # get daily dates
        dates = np.asarray([self.wILI.index[0] + dt.timedelta(days=i) for i in
                            range((self.wILI.index[-1] - self.wILI.index[0]).days + 1)])

        # interpolate weekly to daily
        x = np.linspace(0, 1, self.wILI.shape[0])
        x2 = np.linspace(0, 1, dates.shape[0])
        f = interpolate.interp1d(x, self.wILI.values.squeeze(), kind = 'cubic')

        self.daily_wILI = pd.DataFrame(index=dates, columns=['wILI'], data=f(x2))

        # scale
        self.ili_scaler = MinMaxScaler()
        self.ili_scaler.fit(self.wILI.values.reshape(-1, 1))


        self.wILI = pd.Series(index=self.wILI.index,
                              data=self.ili_scaler.transform(self.wILI.values.reshape(-1, 1)).squeeze())
        self.daily_wILI = pd.Series(index=self.daily_wILI.index,
                                    data=self.ili_scaler.transform(self.daily_wILI.values.reshape(-1, 1)).squeeze())

    def get_queries(self):
        self.queries = pd.read_csv(os.path.join(self.root, 'Queries', 'US_query_data_all_smoothed.csv'), index_col=0,
                                   parse_dates=True)
        # remove duplicate index
        self.queries = self.queries[~self.queries.index.duplicated(keep='first')]
        self.queries = self.queries.sort_index()

        # remove punctuation
        self.queries = self.queries.rename(
            columns={query: query.replace('+', ' ').replace(',', ' ') for query in self.queries.columns})

        # sort queries alphabetically and remove duplicates
        self.queries = self.queries.rename(
            columns={query: ' '.join(sorted(query.split(' '))) for query in self.queries.columns})
        self.queries = self.queries.loc[:, ~self.queries.columns.duplicated()]

        # scale queries to 0-1
        self.query_scaler = MinMaxScaler()
        self.query_scaler.fit(self.queries.values)
        self.queries = pd.DataFrame(index=self.queries.index, columns=self.queries.columns,
                                    data=self.query_scaler.transform(self.queries))

    def get_dates(self):
        self.test_start_date = dt.datetime.strptime(str(self.test_season) + '-W40' + '-1', "%Y-W%W-%w") - dt.timedelta(days=1)+dt.timedelta(weeks=4)
        self.test_dates = np.asarray([self.test_start_date + dt.timedelta(weeks=t) for t in range(1,26)])
        # self.train_dates = pd.date_range(dt.datetime.strftime(self.queries.index[0] + dt.timedelta(days=self.window_size), '%Y/%m/%d')
        #                                  , self.test_dates[0] - dt.timedelta(days=self.gamma))

    def query_to_embedding(self, query):
        if not hasattr(self, 'vectors'):
            self.vectors = pd.read_csv(os.path.join(self.root, 'vectors_unzipped', 'Twitter_word_embeddings_CBOW.csv'),
                                       header=None)
            f = open(os.path.join(self.root, 'vectors_unzipped', 'vocabulary.txt'), "r")
            vocab = f.read()
            vocab = vocab.split('\n')[:-1]
            self.vectors.index = vocab

        query = query.split(' ')
        embedding = []
        for word in query:
            try:
                embedding.append(self.vectors.loc[word].values)
            except:
                embedding.append(np.zeros(self.vectors.shape[1]))
        embedding = np.asarray(embedding).mean(0)[np.newaxis, :]
        return embedding

    def similarity_score(self, embedding, pos=['flu', 'fever', 'flu', 'flu medicine', 'gp', 'hospital'],
                         neg=['bieber', 'ebola', 'wikipedia'], gamma=0.001):
        pos = np.asarray([self.query_to_embedding(p) for p in pos]).squeeze()
        neg = np.asarray([self.query_to_embedding(n) for n in neg]).squeeze()

        pos = cosine_similarity(embedding.reshape(1, -1), pos)
        neg = cosine_similarity(embedding.reshape(1, -1), neg)

        pos = ((pos + 1) / 2).sum()
        neg = ((neg + 1) / 2).sum() + gamma
        return pos / neg

    def query_selection(self):
        # get similarity score
        try:
            scores = pd.read_csv(os.path.join(self.root, 'Similarity_Scores.csv'), index_col=0)
        except:
            query_embeddings = pd.DataFrame(index=self.queries.columns,
                                            data=np.asarray([self.query_to_embedding(query) for query in
                                                             self.queries.columns]).squeeze())
            query_embeddings.to_csv('Data/Query_Embeddings.csv')

            scores = pd.DataFrame(index=self.queries.columns, columns=['similarity'], data=np.asarray(
                [self.similarity_score(embedding) for embedding in query_embeddings.values]))
            scores.to_csv('Data/Similarity_Scores.csv')

        dates = pd.date_range(str(self.data_season - 4) + '/8/23', str(self.data_season + 1) + '/8/23')

        # remove constant frequencies
        self.queries = self.queries.loc[:, self.queries.loc[dates].std() > 0.01]

        scores['correlation'] = pd.DataFrame(index=self.queries.columns,
                                             columns=['correlation'],
                                             data=[pearsonr(self.daily_wILI.loc[dates].squeeze(), q)[0] for q in
                                                   self.queries.loc[dates].values.T])

        scores['correlation'] = (scores['correlation'] + 1) / 2
        scores['correlation'] = scores['correlation'].fillna(scores['correlation'].min())
        scores['similarity'] = scores['similarity'].fillna(scores['similarity'].min())

        if self.selection_method == 'distance':
            scores['distance'] = np.sqrt(np.square(1 - scores / np.tile(scores.max(), (scores.shape[0], 1))).sum(1))
            self.scores = scores.iloc[np.argsort(scores['distance'])]
            self.selected_queries = self.scores.index[:self.n_queries]

        if self.selection_method == 'Bill':
            scores = scores[scores['similarity'] > self.selection_similarity_threshold]
            scores = scores[scores['correlation'] > self.selection_correlation_threshold]
            self.selected_queries = scores.index

if __name__ == "__main__":
    _data = DataConstructor(test_season=2015, data_season=2014, n_queries=200, gamma=28,
                            window_size=42)
    _data()
    inputs, outputs, dates, test_dates = _data.build()
