cd build
docker build . -t tm:1.0
docker run --rm -v ~/contextual_topic_identification:/contextual_topic_identification tm:1.0 --samp_size=1000 --method=LDA --ntopic=5
