FROM python:3.7-slim
COPY ./requirements.txt /
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader averaged_perceptron_tagger
ENTRYPOINT ["python", "-u", "contextual_topic_identification/model/main.py"]
