# FROM continuumio/miniconda3:latest
FROM docker pull nvidia/cuda:11.8.0-runtime-centos7

WORKDIR /opt/workdir
RUN git clone https://github.com/dtak/optimal-summaries-public.git

# manually install cuda
# RUN apt-get update &&
#     apt-get -y upgrade &&
#     apt-get -y install libxml2 &&
#     apt-get -y install build-essential &&
#     apt-get -y install 
# RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
# RUN sh cuda_10.2.89_440.33.01_linux.run

RUN mkdir -p miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
RUN bash miniconda3/miniconda.sh -b -u -p miniconda3
RUN rm -rf miniconda3/miniconda.sh
RUN source miniconda3/bin/activate && conda init

RUN conda create -n optimal-summaries python=3.8.5 ipython notebook

# SHELL ["conda", "run", "-n", "optimal-summaries", "/bin/bash", "-c"]

# RUN conda install cudatoolkit=10.2

RUN pip install -U numpy==1.21.2
RUN pip install -U pandas==1.3.3
RUN pip install -U matplotlib==3.4.2
RUN pip install -U scipy==1.7.1
RUN pip install -U scikit-learn==1.0
RUN pip install h5py
RUN pip install tables

RUN pip install https://download.pytorch.org/whl/cu102/torch-1.5.0-cp38-cp38-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu102/torchvision-0.6.0-cp38-cp38-linux_x86_64.whl

RUN pip install rtpt
RUN pip install tqdm

RUN echo "conda activate optimal-summaries" >> ~/.bashrc
RUN conda config --set auto_activate_base false

#EXPOSE 8090
ENTRYPOINT ["top", "-b"]