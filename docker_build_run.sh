docker build . -t tm:1.0
docker run --rm -v $(pwd):/contextual_topic_identification tm:1.0 --samp_size=3000 --method=LDA_BERT --ntopic=10
