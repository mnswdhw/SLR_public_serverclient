# SLR_public_serverclient


# SLR_server_client


This repository can be used to run the Server Client Split Learning Framework for benchmarking CIFAR-10 on n clients. FOllow the below steps to reproduce the results obtained of Setting-1 of the paper `https://arxiv.org/abs/2303.10624` 

# AWS Setup

Launch the below instance(s) in the same VPC and Subnet of a AWS region. 

* Instance 1: t2.xlarge (16 GB RAM) 
* Instance 2: t2.medium (4 GB RAM) 

# Initializing the instances 

* Each instance needs to initialized with the necessary libraries within a conda environment preferably. 
* Libraries needed

* certifi            2022.12.7
* charset-normalizer 2.1.1
* filelock           3.9.0
* idna               3.4
* Jinja2             3.1.2
* joblib             1.2.0
* MarkupSafe         2.1.2
* mpmath             1.2.1
* networkx           3.0
* numpy              1.24.2
* pandas             2.0.0
* Pillow             9.3.0
* pip                23.0.1
* PuLP               2.7.0
* python-dateutil    2.8.2
* pytz               2023.3
* requests           2.28.1
* scikit-learn       1.2.2
* scipy              1.10.1
* setuptools         67.6.1
* six                1.16.0
* sympy              1.11.1
* threadpoolctl      3.1.0
* torch              2.0.0+cpu
* torchaudio         2.0.1+cpu
* torchvision        0.15.1+cpu
* typing_extensions  4.4.0
* tzdata             2023.3
* urllib3            1.26.13
* wheel              0.40.0

# Running the scripts 

* Server: `python server_manas.py --global_epochs 10 --num_clients 2 --port 9093`
* Client: `python client_manas.py --global_epochs 10 --num_clients 2 --port 9093`

Run the client script n times in the same instance in different terminal windows to simulate n different clients. (This step will be automated later so that in case n is large it is not required to do this step manually)

# Results (Setting-1 Cifar-10) 

* Datapoints Accuracy 
* "50","76.17"
* "150","81.42"
* "250","83.57"
* "350","85"
* "500","85.54"






