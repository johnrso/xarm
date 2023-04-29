MODEL_NAME=2023.04.28/1053_ddt_proprio_variant_0427/0/
DATA_NAME=0427_all_bag_place_conv_euler


rosrun xarm policy_rollout.py \
  --train_config ${DATA_DIR}/${MODEL_NAME}/config.yaml \
  --pol_ckpt ${DATA_DIR}/${MODEL_NAME}/policy_best.ckpt \
  --enc_ckpt ${DATA_DIR}/${MODEL_NAME}/encoder_best.ckpt \
  --conv_config ${DATASET_DIR}/${DATA_NAME}/config.yaml \
  --tag ${MODEL_NAME}
  # --traj ${DATASET_DIR}/${DATA_NAME}/train/none/traj_10.h5
