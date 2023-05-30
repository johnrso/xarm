POLICY_SUFFIX=best
MODEL_PREFIX=0529/0529_0529_bag_two_stream_1900/0
DATA_NAME=place_bag_new/red_black_mg/_conv/multiview_ee_euler/
MODEL_DIR=/data/models/

# if "drawer" is in the data name, then add an argument --safe
if [[ $DATA_NAME == *"drawer"* ]]; then
  echo "Running drawer"
  SAFE="--safe"
else
  echo "Not running drawer"
  SAFE=""
fi

for model_name in "${MODEL_DIR}/${MODEL_PREFIX}"*; do
  # echo "Running model ${model_name}"
  # echo $(basename ${model_name})
  rosrun xarm policy_rollout.py \
    --train_config $model_name/config.yaml \
    --pol_ckpt $model_name/policy_${POLICY_SUFFIX}.ckpt \
    --enc_ckpt $model_name/encoder_${POLICY_SUFFIX}.ckpt \
    --conv_config ${DATASET_DIR}/${DATA_NAME}/config.yaml \
    --tag $(basename ${model_name})  \
    $SAFE \

    # --traj ${DATASET_DIR}/${DATA_NAME}/train/none/traj_10.h5
done
