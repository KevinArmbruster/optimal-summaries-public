FROM continuumio/miniconda3:latest

WORKDIR /opt/workdir
RUN git clone https://github.com/dtak/optimal-summaries-public.git

# manually install cuda
# RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
# RUN sudo sh cuda_10.2.89_440.33.01_linux.run

RUN conda create -n optimal-summaries python=3.8.5 ipython notebook

SHELL ["conda", "run", "-n", "optimal-summaries", "/bin/bash", "-c"]

RUN conda install cudatoolkit=10.2

RUN pip install -U numpy==1.21.2
RUN pip install -U pandas==1.3.3
RUN pip install -U matplotlib==3.4.2
RUN pip install -U scipy==1.7.1
RUN pip install -U scikit-learn==1.0

RUN pip install https://download.pytorch.org/whl/cu102/torch-1.5.0-cp38-cp38-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu102/torchvision-0.6.0-cp38-cp38-linux_x86_64.whl

RUN pip install rtpt

RUN echo "conda activate optimal-summaries" >> ~/.bashrc
RUN conda config --set auto_activate_base false

#EXPOSE 8090
ENTRYPOINT ["top", "-b"]