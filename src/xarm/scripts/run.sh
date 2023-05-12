# MODEL_NAME=0502/no_add_data/0624_0501_proprio_frame
POLICY_SUFFIX=best
MODEL_NAME=0510_red_cup
DATA_NAME=0510_cup_stacking_table_30_conv_multiview_euler
# DATA_NAME=0429_place_bag_tj_black_red_conv_multiview_euler


rosrun xarm policy_rollout.py \
  --train_config ${MODEL_DIR}/${MODEL_NAME}/config.yaml \
  --pol_ckpt ${MODEL_DIR}/${MODEL_NAME}/policy_${POLICY_SUFFIX}.ckpt \
  --enc_ckpt ${MODEL_DIR}/${MODEL_NAME}/encoder_${POLICY_SUFFIX}.ckpt \
  --conv_config ${DATASET_DIR}/${DATA_NAME}/config.yaml \
  --tag ${MODEL_NAME}
  # --traj ${DATASET_DIR}/${DATA_NAME}/train/none/traj_10.h5
