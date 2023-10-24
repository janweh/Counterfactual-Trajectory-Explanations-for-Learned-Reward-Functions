import pickle
import os
import sys
import random

# # read arguments
# if int(sys.argv[1]) == 1:
#     print('1')
# elif int(sys.argv[1]) == 2:
#     print('2')

# print('OUTPUT')
# a = [0,1,2,3]
# with open(os.path.join('interpretability','logs','output.pkl'), 'wb') as f:
#     pickle.dump(a, f)

# # write into text file
# with open(os.path.join('interpretability','logs','outputs.txt'), 'w') as f:
#     f.write('OUTPUT')


# print(os.path.join('interpretability','logs','output.pkl'))

# weights = []
# for i in range(1000):
#     weights.append({'validity': random.uniform(0, 1), 'proximity': random.uniform(0, 1), 'critical_state': random.uniform(0, 1), 'diversity': random.uniform(0, 1), 'realisticness': random.uniform(0, 1), 'sparsity': random.uniform(0, 1)})
# with open(os.path.join('quality_metrics', '1000weights.pkl'), 'wb') as f:
#     pickle.dump(weights, f)

# iterate through folders in 'interpretability\step'
for folder in os.listdir('datasets\\ablations_norm\step'):
    # change the name of the folder
    folder_c = folder.replace('_ending_meetingFalse', '')
    try:
        os.rename(os.path.join('datasets\\ablations_norm\step', folder), os.path.join('datasets\\ablations_norm\step', folder_c))
    except:
        continue
