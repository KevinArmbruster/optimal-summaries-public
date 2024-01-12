FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y apt-transport-https && \
    apt-get install -y git && \
    apt-get install -y curl && \
    apt-get install -y wget && \
    apt-get install -y graphviz && \
    apt-get install -y xdg-utils && \
    apt-get install -y python3-pip

ARG USER_ID
ARG GROUP_ID

# add group, user with group and home, using the host id's
# RUN groupadd -g ${GROUP_ID} karmbruster &&\
#     useradd -l -u ${USER_ID} -g karmbruster karmbruster &&\
#     install -d -m 0755 -o karmbruster -g karmbruster /home/karmbruster

# USER karmbruster
WORKDIR /workdir

RUN git clone https://github.com/KevinArmbruster/optimal-summaries-public.git
RUN git config --global user.name "Kevin Armbruster"
RUN git config --global user.email "KevinArmbruster2013@gmail.com"

RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install ipython "notebook>=5.3" "ipywidgets>=7.5"
RUN pip install numpy pandas matplotlib scipy scikit-learn
RUN pip install h5py tables
RUN pip install rtpt tqdm
RUN pip install seaborn squarify imblearn
RUN pip install optuna plotly kaleido
RUN pip install aeon torchmetrics darts einops captum graphviz

ENTRYPOINT ["tail", "-f", "/dev/null"]

# SETUP INSTRUCTIONS
# docker build --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER}) -t optimal-summaries-env .
# docker run --gpus all -v /home/ml-stud-karmbruster/mimic-iii/data:/workdir/data/mimic-iii -d optimal-summaries-env
