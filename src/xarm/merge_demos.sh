python scripts/merge_demos.py \
  ${DATASET_DIR}/0429_place_bag_tj_black_red_conv_multiview_euler/train/none/ \
  ${DATASET_DIR}/0430_additional_demos_tj_black_red_conv_multiview_euler/train/none/ \
  ${DATASET_DIR}/0503_additional_demos_black_conv_multiview_euler/train/none/ \
  ${DATASET_DIR}/0503_additional_demos_red_conv_multiview_euler/train/none/ \
  --output ${DATASET_DIR}/0507_additional_demos_tj_black_red_conv_multiview_euler/train/ &

python scripts/merge_demos.py \
  ${DATASET_DIR}/0429_place_bag_tj_black_red_conv_multiview_euler/val/none/ \
  ${DATASET_DIR}/0430_additional_demos_tj_black_red_conv_multiview_euler/val/none/ \
  ${DATASET_DIR}/0503_additional_demos_black_conv_multiview_euler/val/none/ \
  ${DATASET_DIR}/0503_additional_demos_red_conv_multiview_euler/val/none/ \
  --output ${DATASET_DIR}/0507_additional_demos_tj_black_red_conv_multiview_euler/val/ &

python scripts/merge_demos.py \
  ${DATASET_DIR}/0429_place_bag_tj_black_red_conv_multiview_ee_euler/train/none/ \
  ${DATASET_DIR}/0430_additional_demos_tj_black_red_conv_multiview_ee_euler/train/none/ \
  ${DATASET_DIR}/0503_additional_demos_black_conv_multiview_ee_euler/train/none/ \
  ${DATASET_DIR}/0503_additional_demos_red_conv_multiview_ee_euler/train/none/ \
  --output ${DATASET_DIR}/0507_additional_demos_tj_black_red_conv_multiview_ee_euler/train/ &

python scripts/merge_demos.py \
  ${DATASET_DIR}/0429_place_bag_tj_black_red_conv_multiview_ee_euler/val/none/ \
  ${DATASET_DIR}/0430_additional_demos_tj_black_red_conv_multiview_ee_euler/val/none/ \
  ${DATASET_DIR}/0503_additional_demos_black_conv_multiview_ee_euler/val/none/ \
  ${DATASET_DIR}/0503_additional_demos_red_conv_multiview_ee_euler/val/none/ \
  --output ${DATASET_DIR}/0507_additional_demos_tj_black_red_conv_multiview_ee_euler/val/ &
