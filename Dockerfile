FROM python:3.7
COPY . /
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader stopwords
CMD ["python", "-u", "./model/main.py"]
