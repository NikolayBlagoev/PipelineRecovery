```
./run.sh no_failure 4 10 500M_config.json 0
./run.sh no_failure 2 10 124M_config.json 0
```

./run.sh no_failure 8 10 1_5B_config.json 0
./run.sh ours-naive 4 16 500M_config.json 0
./run.sh ours-random 4 16 500M_config.json 0
./run.sh ours-grad-avg 2 16 124M_config.json 0
./run.sh ours-grad-avg 8 16 1_5B_config.json 0
./run.sh ours-grad-avg 4 33 500M_config.json 0

./run.sh ours-grad-avg 4 10 500M_config.json 0
./run.sh ours-grad-avg 4 16 500M_config.json 0
524296192
536936448
