FROM mambaorg/micromamba:1.5.8
COPY env/environment.yml /tmp/environment.yml
RUN micromamba create -y -n exp -f /tmp/environment.yml && micromamba clean --all --yes
ENV MAMBA_DEFAULT_ENV=exp
ENV PATH=/opt/conda/envs/exp/bin:$PATH
SHELL ["micromamba","run","-n","exp","/bin/bash","-lc"]
WORKDIR /workspace
CMD ["sleep","infinity"]