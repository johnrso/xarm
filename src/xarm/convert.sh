# MAIN_CONFIG_DIR=dataset/0429_place_bag_tj_black_red_conv_multiview_ee_euler/

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_place_bag_tj_black_red/ \
#   --config-dir $MAIN_CONFIG_DIR \
#   --config-name config &

MAIN_CONFIG_DIR=dataset/0429_place_bag_tj_black_red_conv_multiview_euler/

python scripts/demo_to_gdict.py \
  source_dir=/data/dataset/0429_place_bag_tj_black_red/ \
  --config-dir $MAIN_CONFIG_DIR \
  --config-name config &