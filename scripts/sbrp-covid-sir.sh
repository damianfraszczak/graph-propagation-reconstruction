cd .. && python3 ./sbrp.py \
    --dataset covid-sir \
    --seed 123456789 \
    --data_dir input \
    --device $1 \
    --output output/sbrp-covid-sir.pt
