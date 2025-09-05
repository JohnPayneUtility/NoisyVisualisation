FROM mambaorg/micromamba:1.5.8
COPY env/environment.yml /tmp/environment.yml
RUN micromamba create -y -n exp -f /tmp/environment.yml && micromamba clean --all --yes
SHELL ["micromamba","run","-n","exp","/bin/bash","-lc"]
WORKDIR /workspace
CMD python -c "import mlflow,sys; print(sys.version); print('MLflow', mlflow.__version__)"