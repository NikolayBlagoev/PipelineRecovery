```
./run.sh no_failure 4 10 500M_config.json 0
./run.sh no_failure 4 10 500M_config_gpt.json 0
./run.sh no_failure 2 10 124M_config.json 0
```
./run.sh ours-grad-avg 4 16 500M_config_gpt.json 0


./run.sh no_failure 8 16 1_5B_config.json 100
./run.sh ours-naive 4 10 500M_config.json 0
./run.sh ours-random 4 10 500M_config.json 0
./run.sh ours-grad-avg 2 10 124M_config.json 0
./run.sh ours-grad-avg 8 16 1_5B_config.json 0
./run.sh ours-grad-avg 4 33 500M_config.json 0

./run.sh ours-grad-avg 4 10 500M_config.json 0
./run.sh ours-grad-avg 4 16 500M_config.json 0
./run.sh ours-grad-avg 4 5 500M_config.json 0
./run.sh ours-grad-avg-regularize 4 16 500M_config.json 0
./run.sh ours-grad-avg 4 33 500M_config.json 0

./run.sh ours-zero 4 16 500M_config.json 0


./run_2.sh ours-grad-avg 4 5 500M_config.json 0
./run_2.sh ours-grad-avg 4 10 500M_config.json 0
./run_2.sh ours-grad-avg 4 16 500M_config.json 0
./run_2.sh ours-grad-avg 2 10 124M_config.json 0
524296192
536936448
