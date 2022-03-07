# Silica 2D ML dataset generation
Tutorial on how we generate ML datasets of 2d silica.

## System requirements
The software has been tested on a macOS Big Sur 11.4 with python 3.8.9. 

## Installation guide
The software is tested in a conda enviroment. In order to create the conda enviroment and install the needed package follow the instructions on https://github.com/ComplexityBiosystems/silicanets (this will install our 'silicanets' python library and other needed packages).
In addition, install:
- 'fire' (pip install fire)
- 'imageio' (pip install imageio)

Place the 'silicanets-master' folder in 'Silica-dataset-generation/' path.

## Instructions for generating the initial configurations
The repository 'Example-silica-configuration-generation' contains an example directory with the structure we have used so far for our fixed disorder and variable disorder datasets. To create a new dataset, you just need to replicate the structure of the example directory and follow the instructions.

In order to generate the silica configurations as 'filename.lammps' you have to execute the script
```bash
./generate_initial_configs.sh
```
If everyting goes well, this will fill the `initial_configs` directory.
If we want to change some parameters in the scripts, edit the file `generate_initial_configs.sh`:

```python
  1 #! /bin/zsh
  2
  3 source ~/anaconda3/etc/profile.d/conda.sh
  4 conda activate silicanets
  5
  6
  7 VARMIN="1.00"
  8 VARMAX="1.25"
  9
 10 for i in {0..999}; do
 11     var=`awk -v seed="$RANDOM" -v umin="$VARMIN" -v umax="$VARMAX" 'BEGIN { srand(seed); printf("%.4    f\n", umin + rand()*(umax - umin)) }'`
 12     # do not erase this dummy line!
 13     # otherwise srand seed does not work as intended
 14     dummy=$RANDOM
 15     python get_initial_sample.py \
 16         initial_configs/config_${(l:5::0:)i}.lammps \
 17         --ring_distro_variance $var \
 18         --max_it 1000 > logs/config_${(l:5::0:)i}.log
 19     done
```

the variables `VARMIN` and `VARMAX` set the interval of variance that is being sampled. The value 999 inside the for loop sets how many samples are being generated. Finally, the `--max_it` flag sets the maximum number of swaps in the MCMC procedure. There is no early stop method so you should raise this value carefully.

You can inspect the log files of the MCMC procedure in `/logs`. The last lines of each file show the desired and obtained variances, among other things.
If you the obtained results are too far from the desired values, the you will have to raise the `--maxi_it` parameter. 

## Instructions for generating silica images

The repository 'Example-silica-image-generation' contains an example directory to generate silica images from configurations. An example of the starting point configuration is 'stretched_configs/run_00001' which contains the lammps file for the initial and broken configuration of silica.

In order to generate the silica image you should run:

```bash
python generate_ML_dataset.py parse_dataset  \
  --glob_to_runs "stretched_configs/run_0000*" \
  --path_to_images_dir ml_dataset/images \
  --output_metadata ml_dataset/metadata.csv \
  --num_translations 16
```
If you want to generate only a subset of images, you can play with the `--glob_to_runs` flag which does pattern matching on the input stretched configurations. When the script finishes, you should find all your images in `ml_dataset/images` and the metadata file in `ml_dataset/metadata.csv`.

The generated images will be stored into `ml_dataset/images`. The script will also generate a metadata CSV file, which tells you which image corresponds to which configuration and under which data-augmentation parameters (reflections and translations). We generate 64 images per configuration (2x up-down flip, 2x left-right flip, 16x translation). 

