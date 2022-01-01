exp=$1
others=$2

python -m hyperbox.run \
hydra.searchpath=[/home/comp/18481086/code/hyperbox_app/medmnist/configs] \
experiment=$exp \
$others
