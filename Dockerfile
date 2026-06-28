FROM continuumio/anaconda3:latest

ARG env_name=MA-DDI

SHELL ["/bin/bash", "-c"]

WORKDIR /media/ST-18T/Ma/image

COPY . ./

RUN conda create -n $env_name python==3.8.17 \
&& source deactivate \
&& conda activate $env_name \
&& conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge \
&& pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
&& pip install torch_geometric==2.2.0
&& pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

RUN echo "source activate $env_name" > ~/.bashrc
ENV PATH /opt/conda/envs/$env_name/bin:$PATH


CMD ["/bin/bash","inference.sh"]
