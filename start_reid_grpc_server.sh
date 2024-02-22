python3 reid/server_grpc.py \
    --port 2202 \
    --device cuda:3 \
    --config_file configs/reid/AIC/bagtricks_R50.yml \
    --weights_path weights/market_aic_bot_R50.pth \
    --batch_size 1
