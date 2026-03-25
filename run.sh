docker run -it --rm \
    --runtime=nvidia \
    -v $(pwd)/data_off:/kairos/data_off \
    -e WANDB_API_KEY="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" \
    kairos-agent:latest
