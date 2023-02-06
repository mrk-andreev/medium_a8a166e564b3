FROM ubuntu:23.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

RUN apt-get update
RUN apt-get install -y r-base
RUN apt-get install -y python3 python3-pip
RUN R -e "install.packages(\"reticulate\")"
RUN R -e "install.packages(\"microbenchmark\")"
RUN pip install scikit-learn==1.2.1 pandas==1.5.3
COPY train.py .
COPY predict_pipeline.py .
