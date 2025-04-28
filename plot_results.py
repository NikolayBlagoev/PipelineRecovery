import matplotlib.pyplot as plt
import math
import numpy as np
plt.figure(figsize=(16,10))
plt.locator_params(axis='x', nbins=10)
def smooth_func(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def plot_fl(fl,label, validation = False, pad = [], flag = False, max_el = -1, show_failures = False, smooth = 1):
    start = False
    validation_loss = [] + pad
    training_loss = []
    tmp = 0
    prev_checkpoint = 0
    actual_iteration = 0
    actual_run = []
    failures = []
    with open(fl,"r") as fd:
        for ln in fd.readlines():
            
            if "Iteration failure probability" in ln:
                start = True
                continue
            if not start:
                continue
            if "SAVED" in ln:
                continue
            if "failure" in ln:
                if flag:
                    # print(actual_iteration, "returinging to ", prev_checkpoint)
                    
                    actual_iteration = prev_checkpoint
                to_add = int(ln.split(" ")[1])
                node = int(ln.split(" ")[-1].strip())
                # print(node)
                # if node != 0 and node != 5:
                #     continue
                if to_add > max_el:
                    break
                failures.append(to_add)
                

                continue
            if "time" in ln:
                continue
            if "SAVING" in ln:
                # print(ln)
                prev_checkpoint = actual_iteration
                # print(label,prev_checkpoint)
                continue
            if "NORMAL" in ln and validation:
                
                
                validation_loss.append(float(ln.strip().split(" ")[3].strip()))
                if label != "Baseline" and validation_loss[-1] < 2.9:
                    break
                continue
            # print(ln)
            if "VALIDATION" in ln or "NORMAL" in ln:
                continue

            # print(ln)

            try:
                # print(ln)
                actual_run.append(actual_iteration)
                actual_iteration += 1
                training_loss.append(float(ln.split(" ")[1].strip()))
                
            except ValueError:
                continue
            except IndexError:
                continue
    # print(arr1)
    if validation:
        max_el = max_el // 500
    ret = training_loss
    if validation:
        ret = validation_loss
    
    print(label,len(actual_run))
    if flag:
        if validation:
            tmp = []
            for i in range(0,len(actual_run)//500):
                # print(i)
                i = actual_run[i*500] / 500
                fl = math.floor(i)
                cl = math.ceil(i)
                alpha = i - fl
                tmp.append((1-alpha)*validation_loss[fl] + alpha * validation_loss[cl])
            
        else:
            tmp = []
            for i in range(0,max_el):
                tmp.append(training_loss[actual_run[i]])

        ret = tmp
    ret = ret[:max_el]
    ret = smooth_func(ret,smooth)
    plt.plot(ret,label=label)
    
    if validation:
        # plt.xticks(list(range(len(ret))),list(map(lambda el: el*500, list(range(len(ret))))))
        if show_failures:
            plt.vlines(list(map(lambda el: el/500,failures)),ymin=0,ymax=10,color=(1,0,0,0.1))
    else:
        if show_failures:
            plt.vlines(failures,ymin=0,ymax=10,color=(1,0,0,0.1))

MAX_EL = 100000
validate = True
show_failures = False
smooth = 1
plot_fl("results/medium_naive_16/out0.txt", "Naive copy",max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
plot_fl("results/medium_baseline_16/out0.txt", "Baseline",flag=True,max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
plot_fl("results/medium_gradavg_16/out0.txt", "Ours",max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
plot_fl("results/medium_gradavg_10/out0.txt", "Ours 10",max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
plot_fl("results/medium_gradavg_33/out0.txt", "Ours 33",max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
plot_fl("results/medium_baseline_16/out0.txt", "No Failure",max_el=MAX_EL,validation=validate, show_failures=show_failures, smooth = smooth)
# plot_fl("results/to_send_small_no_fault/out0.txt", "Baseline")
# plot_fl("results/to_send_grad_avg_16_small/out0.txt", "ours", pad=[10.04,10.04,10.04,10.04,10.04,10.04])

plt.legend()
plt.savefig("medium_validation_results_recovery.pdf")
plt.show()