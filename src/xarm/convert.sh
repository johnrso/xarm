# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_.25/ \
#   rotation_mode=euler --config-name no_ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_.25/ \
#   rotation_mode=euler --config-name ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_.5/ \
#   rotation_mode=euler --config-name no_ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_.5/ \
#   rotation_mode=euler --config-name ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_.75/ \
#   rotation_mode=euler --config-name no_ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_.75/ \
#   rotation_mode=euler --config-name ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_1/ \
#   rotation_mode=euler --config-name no_ee &

# python scripts/demo_to_gdict.py \
#   source_dir=/data/dataset/0429_test_smoothing_alpha_1/ \
#   rotation_mode=euler --config-name ee &


MAIN_CONFIG_DIR=dataset/0429_place_bag_tj_black_red_conv_multiview_ee_euler/

python scripts/demo_to_gdict.py \
  source_dir=/data/dataset/0503_additional_demos_black/ \
  --config-dir $MAIN_CONFIG_DIR \
  --config-name config &

python scripts/demo_to_gdict.py \
  source_dir=/data/dataset/0503_additional_demos_red/ \
  --config-dir $MAIN_CONFIG_DIR \
  --config-name config &

MAIN_CONFIG_DIR=dataset/0429_place_bag_tj_black_red_conv_multiview_euler/

python scripts/demo_to_gdict.py \
  source_dir=/data/dataset/0503_additional_demos_black/ \
  --config-dir $MAIN_CONFIG_DIR \
  --config-name config &

python scripts/demo_to_gdict.py \
  source_dir=/data/dataset/0503_additional_demos_red/ \
  --config-dir $MAIN_CONFIG_DIR \
  --config-name config &