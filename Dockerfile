FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq \
    && apt-get install -y r-base python3 python3-pip
RUN R -e "install.packages(\"reticulate\")"
RUN R -e "install.packages(\"microbenchmark\")"
RUN pip install scikit-learn==1.2.1 pandas==1.5.3
