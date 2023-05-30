python src/merge_datasets.py \
  ${DATASET_DIR}/drawer/white_bot_screwdriver \
  ${DATASET_DIR}/drawer/white_top_screwdriver_2 \
  ${DATASET_DIR}/drawer/green_bot_screwdriver \
  ${DATASET_DIR}/drawer/pink_mid_screwdriver \
  ${DATASET_DIR}/drawer/pink_top_screwdriver \
  --output ${DATASET_DIR}/drawer/pwg_screw_1 &

python src/merge_datasets.py \
  ${DATASET_DIR}/drawer/white_mid_screwdriver \
  ${DATASET_DIR}/drawer/green_top_screwdriver \
  ${DATASET_DIR}/drawer/pink_bot_screwdriver \
  ${DATASET_DIR}/drawer/clear_bot_screwdriver \
  --output ${DATASET_DIR}/drawer/pwg_screw_2 &