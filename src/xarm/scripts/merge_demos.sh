python src/merge_demos.py \
  ${DATASET_DIR}/drawer/green_top_screwdriver/_conv/multiview_ee_euler/train/none/ \
  ${DATASET_DIR}/drawer/green_bot_screwdriver/_conv/multiview_ee_euler/train/none/ \
  --output ${DATASET_DIR}/drawer/green_all_screw/_conv/multiview_ee_euler/train/ &

python src/merge_demos.py \
  ${DATASET_DIR}/drawer/green_top_screwdriver/_conv/multiview_ee_euler/val/none/ \
  ${DATASET_DIR}/drawer/green_bot_screwdriver/_conv/multiview_ee_euler/val/none/ \
  --output ${DATASET_DIR}/drawer/green_all_screw/_conv/multiview_ee_euler/val/ &