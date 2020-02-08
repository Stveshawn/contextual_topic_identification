FROM python:3.7-slim
# COPY ./model /model
COPY ./requirements.txt /
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader averaged_perceptron_tagger
CMD ["python", "-u", "contextual_topic_identification/model/main.py", "--samp_size=100", "--method=TFIDF"]
