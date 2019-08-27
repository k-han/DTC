path="data/"
mkdir -p $path
cd $path

wget http://www.robots.ox.ac.uk/~vgg/research/DTC/data/experiments.zip

unzip experiments.zip && rm experiments.zip

cd ../
