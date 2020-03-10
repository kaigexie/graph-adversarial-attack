#!/bin/bash

# dropbox=../../dropbox
dropbox=dropbox

min_n=90
max_n=100
p=0.02
min_c=1
max_c=3
base_lv=2
data_folder=$dropbox/data/components
save_fold=nodes-${min_n}-${max_n}-p-${p}-c-${min_c}-${max_c}-lv-${base_lv}
base_model_dump=$dropbox/scratch/results/graph_classification/components/$save_fold/epoch-best

lr=0.001
max_lv=5
frac_meta=0.1

# output_base=scratch/results/graph_classification/components/$save_fold
output_base=code/graph_attack/scratch/results/graph_classification/components/$save_fold




output_root=$output_base/lv-${max_lv}-frac-${frac_meta}

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

# for (( ; ; ))
# do
#     echo ""
#     echo "------------------------------------------------------"
#     echo "Program starts."
#     echo "Filename of input graph: [ hit CTRL+C to stop] [E.g. ncomp-3-nrange-90-100-n_graph-5000-p-0.02.pkl]"
#     read graph_fname
# python -W ignore get_attack_result.py \
#     -data_folder $data_folder \
#     -save_dir $output_root \
#     -max_n $max_n \
#     -min_n $min_n \
#     -max_lv $max_lv \
#     -frac_meta $frac_meta \
#     -min_c $min_c \
#     -max_c $max_c \
#     -n_graphs 5000 \
#     -er_p $p \
#     -learning_rate $lr \
#     -base_model_dump $base_model_dump \
#     -logfile $output_root/log.txt \
#     -graph_fname $graph_fname \
#     -graph_type 3 \
#     -num_graph 200 \
#     -display_mode 1 \
#     $@
#     echo "Program ends."
#     echo "------------------------------------------------------"
# done

echo ""
echo "------------------------------------------------------"
echo "Program starts."

python -W ignore code/graph_attack/get_attack_result.py \
    -data_folder $data_folder \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -max_lv $max_lv \
    -frac_meta $frac_meta \
    -min_c $min_c \
    -max_c $max_c \
    -n_graphs 5000 \
    -er_p $p \
    -learning_rate $lr \
    -base_model_dump $base_model_dump \
    -logfile $output_root/log.txt \
    -graph_fname ncomp-3-nrange-90-100-n_graph-5000-p-0.02.pkl \
    -graph_type 3 \
    -num_graph 200 \
    -display_mode 1 \
    $@

echo "Program ends."
echo "------------------------------------------------------"
