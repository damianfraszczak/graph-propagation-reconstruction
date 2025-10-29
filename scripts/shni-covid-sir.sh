cd .. && python3 ./shni.py \
    --dataset covid-sir \
    --seed 123456789 \
    --data_dir input \
    --device $1 \
    --output output/shni-covid-sir.pt
