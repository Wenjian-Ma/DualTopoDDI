FROM continuumio/anaconda3:latest

ARG env_name=MA-DDI

SHELL ["/bin/bash", "-c"]

WORKDIR /media/ST-18T/Ma/image

COPY . ./

RUN conda create -n $env_name python==3.8.17 \
&& source deactivate \
&& conda activate $env_name \
&& conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge \
&& pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
&& pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu117.html

RUN echo "source activate $env_name" > ~/.bashrc
ENV PATH /opt/conda/envs/$env_name/bin:$PATH


CMD ["/bin/bash","inference.sh"]