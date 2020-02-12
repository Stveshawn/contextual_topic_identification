docker build . -t tm_test:1.0
docker run --rm -v $(pwd):/contextual_topic_identification tm_test:1.0 --samp_size=1000 --method=TFIDF --ntopic=10 --fpath=/contextual_topic_identification/data/test.csv
# docker run --rm -v $(pwd):/contextual_topic_identification tm_test:1.0 --samp_size=1000 --method=LDA --ntopic=10 --fpath=/contextual_topic_identification/data/test.csv
# docker run --rm -v $(pwd):/contextual_topic_identification tm_test:1.0 --samp_size=1000 --# method=BERT --ntopic=10 --fpath=/contextual_topic_identification/data/test.csv
# docker run --rm -v $(pwd):/contextual_topic_identification tm_test:1.0 --samp_size=1000 --method=LDA_BERT --ntopic=10 --fpath=/contextual_topic_identification/data/test.csv
