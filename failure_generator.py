import random
random.seed(21)
hourly_failure_rate = 33
p_fail = 0

if hourly_failure_rate == 10:
    p_fail = 0.002
elif hourly_failure_rate == 16:
    p_fail == 0.004
elif hourly_failure_rate == 33:
    p_fail = 0.01


stages = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [9,10,11],
    [12,13,14],
    [15,16,17],
    [18,19,20]
]
failures = {}

for itr in range(33000):
    # print("------")
    for i,s in enumerate(stages):
        faults = 0
        stage_failure = 2 if random.random() < 1 - (1 - p_fail)**len(s) else 1
        
        
        for nd in s:
            if nd not in failures:
                failures[nd] = []
            if random.random() < 2 * p_fail and i > 0 and itr > 0:
                faults += 1
                failures[nd].append(random.randint(0,3))
            else:
                failures[nd].append(-1)
            
                
        if faults == 3:
            print("STAGE GONE",i,itr)
       
    


# print(failures)
