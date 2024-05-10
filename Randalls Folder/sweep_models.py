import subprocess
from tqdm import tqdm


# Known parameters
OUTPUT_LABELS = ['all', 0, 1, 2, 3, 4, 5, 6]
SUBSET_TYPES = ['random', 'all', 'filter1']
APPLY_LOGS = [True, False]
MODEL_TYPES = [1, 2, 3]
DROPOUT_TYPES = [1, 2, 3]
VALIDATION_TYPES = ['random']
CRITERIONS = ['mse']
OPTIMIZERS = ['SGD', 'Adam']

# Define the range of values for each parameter
SKIP_TO = 0 # Where to start in the parameter sweep
output_labels = ['all']#, 0, 1, 2, 3, 4, 5, 6]
subset_type = ['filter1']
apply_log = [True]#[True, False]   # Can't really tell the difference
random_seed = [42]
sample_size = [0.05]
test_size = [0.2]
batch_size = [32, 128, 1024]
num_epochs = [100]
model_type = [1, 2, 4, 6] # 1 and 2 are perceptron models, 3 is a CNN
dropout_type = [1]
validation_type = ['random']
criterion = ['mse_seperated']
optimizer = ['SGD']
lr = [5e-4]
momentum = [0.9]
REPORT_BASE_NAME = "sigmoid_end_report7"
#report_name = ['report_test']
# report_title = ['"Test Report"']
REPORT_TITLE = '"Last layer sigmoid Report"'


# Write parameters to a text file
with open(f'/home/rlfowler/Documents/research/tfo_inverse_modelling/results/{REPORT_BASE_NAME}_params.txt', 'w') as file:
    file.write('output_labels = ' + str(output_labels) + '\n')
    file.write('subset_type = ' + str(subset_type) + '\n')
    file.write('apply_log = ' + str(apply_log) + '\n')
    file.write('random_seed = ' + str(random_seed) + '\n')
    file.write('sample_size = ' + str(sample_size) + '\n')
    file.write('test_size = ' + str(test_size) + '\n')
    file.write('batch_size = ' + str(batch_size) + '\n')
    file.write('num_epochs = ' + str(num_epochs) + '\n')
    file.write('model_type = ' + str(model_type) + '\n')
    file.write('dropout_type = ' + str(dropout_type) + '\n')
    file.write('validation_type = ' + str(validation_type) + '\n')
    file.write('criterion = ' + str(criterion) + '\n')
    file.write('optimizer = ' + str(optimizer) + '\n')
    file.write('lr = ' + str(lr) + '\n')
    file.write('momentum = ' + str(momentum) + '\n')
    file.write('REPORT_BASE_NAME = ' + str(REPORT_BASE_NAME) + '\n')
    file.write('REPORT_TITLE = ' + str(REPORT_TITLE) + '\n')



# Iterate over the parameter combinations
i = 0
total = len(output_labels) * len(subset_type) * len(apply_log) * len(random_seed) * len(sample_size) * len(test_size) * len(batch_size) * len(num_epochs) * len(model_type) * len(dropout_type) * len(validation_type) * len(criterion) * len(optimizer) * len(lr) * len(momentum)
with tqdm(total=total) as pbar:
    for ol in output_labels:
        for st in subset_type:
            for al in apply_log:
                for rs in random_seed:
                    for ss in sample_size:
                        for ts in test_size:
                            for bs in batch_size:
                                for ne in num_epochs:
                                    for mt in model_type:
                                        for do in dropout_type:
                                            for vt in validation_type:
                                                for c in criterion:
                                                    for o in optimizer:
                                                        for l in lr:
                                                            for m in momentum:
                                                                if i < SKIP_TO:
                                                                    pbar.update(1)
                                                                    i += 1
                                                                    pbar.refresh()
                                                                    continue
                                                                
                                                                # Construct the command
                                                                cmd = f"python Randalls\ Folder/FC_model.py "
                                                                if ol != None:
                                                                    cmd += f"-ol {ol} "
                                                                if st != None:
                                                                    cmd += f"-st {st} "
                                                                if al != None:
                                                                    cmd += f"-al {al} "
                                                                if rs != None:
                                                                    cmd += f"-rs {rs} "
                                                                if ss != None:
                                                                    cmd += f"-ss {ss} "
                                                                if ts != None:
                                                                    cmd += f"-ts {ts} "
                                                                if bs != None:
                                                                    cmd += f"-bs {bs} "
                                                                if ne != None:
                                                                    cmd += f"-ne {ne} "
                                                                if mt != None:
                                                                    cmd += f"-mt {mt} "
                                                                if do != None:
                                                                    cmd += f"-do {do} "
                                                                if vt != None:
                                                                    cmd += f"-vt {vt} "
                                                                if c != None:
                                                                    cmd += f"-c {c} "
                                                                if o != None:
                                                                    cmd += f"-o {o} "
                                                                if l != None:
                                                                    cmd += f"-l {l} "
                                                                if m != None:
                                                                    cmd += f"-m {m} "
                                                                if REPORT_BASE_NAME != None:
                                                                    cmd += f"-rn {REPORT_BASE_NAME}_{i} "
                                                                if REPORT_TITLE != None:
                                                                    cmd += f"-rt {REPORT_TITLE}"
  
                                                                i += 1
                                                                if i == 1:
                                                                    cmd += " > output_log.txt" # Create a new file
                                                                else:
                                                                    cmd += " >> output_log.txt" # Append to the file

                                                                # Execute the command
                                                                #print(cmd)
                                                                subprocess.run(cmd, shell=True)
                                                                pbar.update(1)

print("All done!")