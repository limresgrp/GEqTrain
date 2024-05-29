# conda create --name geqtrain python=3.10
conda install pytorch==1.13 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch-scatter
pip install -e .