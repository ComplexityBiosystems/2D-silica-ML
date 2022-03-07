#! /bin/zsh 

source ~/anaconda3/etc/profile.d/conda.sh
conda activate silicanets


VARMIN="1.00"
VARMAX="1.25"

for i in {4..999}; do
    var=`awk -v seed="$RANDOM" -v umin="$VARMIN" -v umax="$VARMAX" 'BEGIN { srand(seed); printf("%.4f\n", umin + rand()*(umax - umin)) }'`
    # do not erase this dummy line!
    # otherwise srand seed does not work as intended
    dummy=$RANDOM
    echo "var:",$var
    python get_initial_sample.py \
        initial_configs/config_${(l:5::0:)i}.lammps \
        --ring_distro_variance $var \
        --max_it 1000 > logs/config_${(l:5::0:)i}.log
    done
