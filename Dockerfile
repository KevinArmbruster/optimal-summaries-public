FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y git && \
    apt-get install -y curl && \
    apt-get install -y wget 
    # && \
    # apt-get install -y software-properties-common && \
    # add-apt-repository ppa:deadsnakes/ppa && \
    # apt-get install -y python3.8


WORKDIR /workdir

RUN git clone https://github.com/KevinArmbruster/optimal-summaries-public.git
RUN git config --global user.name "Kevin Armbruster"
RUN git config --global user.email "KevinArmbruster2013@gmail.com"


RUN pip install --upgrade pip
RUN pip install ipython notebook
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install h5py
RUN pip install tables
RUN pip install rtpt
RUN pip install tqdm

# RUN echo "conda activate optimal-summaries" >> ~/.bashrc
# RUN conda config --set auto_activate_base false


#EXPOSE 8090
ENTRYPOINT ["top", "-b"]

# SETUP INSTRUCTIONS
# docker build -t optimal-summaries-env .
# docker run --name optimal-summaries-env3 -v /home/karmbruster/mimic-iii/physionet.org/export:/workdir/data -d optimal-summaries-env
