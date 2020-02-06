from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# define model object
class Topic_Model:
    def __init__(self, k=10, method='TFIDF'):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec_lda = None
        self.vec_bert = None
        self.vec_tfidf = None
        self.vec_ldabert = None
        self.vec_ldabert_latent = None
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.sentences = None
        self.token_lists = None

    # # turn our tokenized documents into a id <-> term dictionary
    # dictionary = corpora.Dictionary(rws_processed)

    # # # convert tokenized documents into a document-term matrix
    # corpus = [dictionary.doc2bow(text) for text in rws_processed]

    def preprocess(self, docs, inplace=None, samp_size=None):
        """
        Preprocess the data
        """
        if inplace is None:
            inplace = True
        if not samp_size:
            samp_size = 100

        n_docs = len(docs)
        if inplace:
            self.sentences = []  # sentence level preprocessed
            self.token_lists = []  # word level preprocessed
            self.idx_in = []  # index of sample selected
            #             samp = np.random.choice(n_docs, 100)
            samp = list(range(100))
            for i, idx in enumerate(samp):
                sentence = preprocess_sent(docs[idx])
                token_list = preprocess_word(sentence)
                if token_list:
                    self.idx_in.append(idx)
                    self.sentences.append(sentence)
                    self.token_lists.append(token_list)
                print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
        else:
            sentences = []  # sentence level preprocessed
            token_lists = []  # word level preprocessed
            idx_in = []  # index of sample selected
            samp = list(range(100))
            #             samp = np.random.choice(n_docs, 100)
            for i, idx in enumerate(samp):
                sentence = preprocess_sent(docs[idx])
                token_list = preprocess_word(sentence)
                if token_list:
                    idx_in.append(idx)
                    sentences.append(sentence)
                    token_lists.append(token_list)
                print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')

            return sentences, token_lists, idx_in

    def vectorize(self, docs, method=None):
        """
        Get vecotr representations from selected methods
        """
        # Default method
        if method is None:
            method = self.method
        # if not preprocess yet
        if not self.sentences:
            # if self.docs is not docs:
            self.preprocess(docs)

        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(self.token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in self.token_lists]

        if method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            self.vec_tfidf = tfidf.fit_transform(self.sentences)
            print('Getting vector representations for TF-IDF. Done!')


        elif method == 'LDA':
            print('Getting vector representations for LDA ...')
            self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                            passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            self.vec_lda = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            self.vec_bert = np.array(model.encode(self.sentences, show_progress_bar=True))
            print('Getting vector representations for BERT. Done!')
        #         elif method == 'LDA_BERT':
        else:
            if not self.vec_lda:
                self.vectorize(docs, method='LDA')
            if not self.vec_bert:
                self.vectorize(docs, method='BERT')

            self.vec_ldabert = np.c_[self.vec_lda * self.gamma, self.vec_bert]
            AE = Autoencoder()
            print('Fitting Autoencoder ...')
            AE.fit(self.vec_ldabert)
            self.vec_ldabert_latent = AE.encoder.predict(self.vec_ldabert)
            print('Fitting Autoencoder Done!')

    def fit(self, docs, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans
        # Preprocess data if not yet
        if not self.sentences:
            # if self.docs is not docs:
            self.preprocess(docs)

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(self.token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in self.token_lists]

        ####################################################
        #### Getting ldamodel or vector representations ####
        ####################################################

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)
                print('Fitting LDA Done!')
        else:
            self.vectorize(docs, method)
            self.cluster_model = m_clustering(self.k)
            if method == 'TFIDF':
                self.cluster_model.fit(self.vec_tfidf)
            elif method == 'BERT':
                self.cluster_model.fit(self.vec_bert)
            else:
                self.cluster_model.fit(self.vec_ldabert_latent)

    def predict(self, new_docs, out_of_sample=None):
        """
        Predict topics for new_documents
        """
        # Default as False
        out_of_sample = out_of_sample is not None
        if out_of_sample:
            sentences, token_lists, idx_in = self.preprocess(new_docs, inplace=False)
            #         # turn tokenized documents into a id <-> term dictionary
            #         dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
        else:
            corpus = self.corpus

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            if self.method == 'TFIDF':
                lbs = self.cluster_model.predict(self.vec_tfidf)
            elif self.method == 'BERT':
                lbs = self.cluster_model.predict(self.vec_bert)
            else:
                lbs = self.cluster_model.predict(self.vec_ldabert_latent)
        return lbs