# resfoc
Improving seismic image focusing via residual migration and deep neural networks  

## Usage and build
If you are interested in what you can do with the code in this repository,
please look at the Jupyter notebooks found within the `notebooks`
directory.

If you would like to run the notebooks or the code, I would recommend building
a Docker image using the `Dockerfile`. This will first require the installation of docker
and then [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

With docker and nvidia-docker installed, you can build a docker image with the command:

`docker build -t "resfoc:latest" .`,

assuming that you are in the same directory as `Dockerfile`.

You can then run the docker image with:

`docker run -p 8888:8888 --gpus all resfoc:latest`,

which will start a Jupyter notebook and will provide a local URL that you can put
into your browser to run the notebooks.

If you would prefer not to use docker, then the `Dockerfile` shows exactly all of the 
steps and required software to install and run the notebooks.
