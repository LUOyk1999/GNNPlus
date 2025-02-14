counter=$1

for gnn in gcn gine gatedgcn
do

export CUDA_VISIBLE_DEVICES=$((counter % 8))

python main.py --cfg configs/$gnn/$2.yaml --repeat $3 seed 0 

done
