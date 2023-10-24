import pickle
import numpy as np
from tabulate import tabulate

path = "datasets\\ablations_norm\\a_ending_prob0.15_num_deviations2_1000\statistics\qc_statistics.pkl"
with(open(path, 'rb')) as f:
    qc_stats = pickle.load(f)

qc_val_spears, qc_prox_spears, qc_crit_spears, qc_div_spears, qc_real_spears, qc_spar_spears, val_prox_spears, val_crit_spears, val_div_spears, prox_crit_spears, prox_div_spears, crit_div_spears = [], [], [], [], [], [], [], [], [], [], [], []
qc_val_pears, qc_prox_pears, qc_crit_pears, qc_div_pears, qc_real_pears, qc_spar_pears, val_prox_pears, val_crit_pears, val_div_pears, prox_crit_pears, prox_div_pears, crit_div_pears = [], [], [], [], [], [], [], [], [], [], [], []
pos_val, pos_prox, pos_crit, pos_div, pos_real, pos_spar = [], [], [], [], [], []
chosen_sum, chosen_val, chosen_prox, chosen_crit, chosen_div, chosen_real, chosen_spar = [], [], [], [], [], [], []
highest_val, highest_prox, highest_crit, highest_div, highest_real, highest_spar = [], [], [], [], [], []
lowest_val, lowest_prox, lowest_crit, lowest_div, lowest_real, lowest_spar = [], [], [], [], [], []
range_val, range_prox, range_crit, range_div, range_real, range_spar = [], [], [], [], [], []
contribution_val, contribution_prox, contribution_crit, contribution_div, contribution_real, contribution_spar = [], [], [], [], [], []

for i in range(len(qc_stats)):
    # load all the correlations saved as this dictionary
    # {'qc-validity': qc_val_spear, 'qc-proximity': qc_prox_spear, 'qc-critical_state': qc_crit_spear, 'qc-diversity': qc_div_spear, 'validity-proximity': val_prox_spear, 'validity-critical_state': val_crit_spear, 'validity-diversity': val_div_spear, 'proximity-critical_state': prox_crit_spear, 'proximity-diversity': prox_div_spear, 'critical_state-diversity': crit_div_spear}
    qc_val_spears.append(qc_stats[i]['spear_correlations']['qc-validity'])
    qc_prox_spears.append(qc_stats[i]['spear_correlations']['qc-proximity'])
    qc_crit_spears.append(qc_stats[i]['spear_correlations']['qc-critical_state'])
    qc_div_spears.append(qc_stats[i]['spear_correlations']['qc-diversity'])
    qc_real_spears.append(qc_stats[i]['spear_correlations']['qc-realisticness'])
    qc_spar_spears.append(qc_stats[i]['spear_correlations']['qc-sparsity'])
    # val_prox_spears.append(qc_stats[i]['spear_correlations']['validity-proximity'])
    # val_crit_spears.append(qc_stats[i]['spear_correlations']['validity-critical_state'])
    # val_div_spears.append(qc_stats[i]['spear_correlations']['validity-diversity'])
    # prox_crit_spears.append(qc_stats[i]['spear_correlations']['proximity-critical_state'])
    # prox_div_spears.append(qc_stats[i]['spear_correlations']['proximity-diversity'])
    # crit_div_spears.append(qc_stats[i]['spear_correlations']['critical_state-diversity'])


    qc_val_pears.append(qc_stats[i]['pear_correlations']['qc-validity'])
    qc_prox_pears.append(qc_stats[i]['pear_correlations']['qc-proximity'])
    qc_crit_pears.append(qc_stats[i]['pear_correlations']['qc-critical_state'])
    qc_div_pears.append(qc_stats[i]['pear_correlations']['qc-diversity'])
    qc_real_pears.append(qc_stats[i]['pear_correlations']['qc-realisticness'])
    qc_spar_pears.append(qc_stats[i]['pear_correlations']['qc-sparsity'])
    # val_prox_pears.append(qc_stats[i]['pear_correlations']['validity-proximity'])
    # val_crit_pears.append(qc_stats[i]['pear_correlations']['validity-critical_state'])
    # val_div_pears.append(qc_stats[i]['pear_correlations']['validity-diversity'])
    # prox_crit_pears.append(qc_stats[i]['pear_correlations']['proximity-critical_state'])
    # prox_div_pears.append(qc_stats[i]['pear_correlations']['proximity-diversity'])
    # crit_div_pears.append(qc_stats[i]['pear_correlations']['critical_state-diversity'])

    pos_val.append(qc_stats[i]['perc_positions']['validity'])
    pos_prox.append(qc_stats[i]['perc_positions']['proximity'])
    pos_crit.append(qc_stats[i]['perc_positions']['critical_state'])
    pos_div.append(qc_stats[i]['perc_positions']['diversity'])
    pos_real.append(qc_stats[i]['perc_positions']['realisticness'])
    pos_spar.append(qc_stats[i]['perc_positions']['sparsity'])

    v = qc_stats[i]['chosen_values']['validity']
    chosen_val.append(v)
    p = qc_stats[i]['chosen_values']['proximity']
    if p != float('inf'):
        chosen_prox.append(p)
    d = qc_stats[i]['chosen_values']['diversity']
    chosen_div.append(d)
    c = qc_stats[i]['chosen_values']['critical_state']
    chosen_crit.append(c)
    r = qc_stats[i]['chosen_values']['realisticness']
    chosen_real.append(r)
    s = qc_stats[i]['chosen_values']['sparsity']
    chosen_spar.append(s)
    chosen_s = v+p+d+c+r+s
    chosen_sum.append(chosen_s)

    contribution_val.append(v/chosen_s)
    if p != float('inf'):
        contribution_prox.append(p/chosen_s)
    contribution_crit.append(c/chosen_s)
    contribution_div.append(d/chosen_s)
    contribution_real.append(r/chosen_s)
    contribution_spar.append(s/chosen_s)

    highest_val.append(qc_stats[i]['highest_values']['validity'])
    x = qc_stats[i]['highest_values']['proximity']
    if x != float('inf'):
        highest_prox.append(qc_stats[i]['highest_values']['proximity'])
    highest_crit.append(qc_stats[i]['highest_values']['critical_state'])
    highest_div.append(qc_stats[i]['highest_values']['diversity'])
    highest_real.append(qc_stats[i]['highest_values']['realisticness'])
    highest_spar.append(qc_stats[i]['highest_values']['sparsity'])

    lowest_val.append(qc_stats[i]['lowest_values']['validity'])
    lowest_prox.append(qc_stats[i]['lowest_values']['proximity'])
    lowest_crit.append(qc_stats[i]['lowest_values']['critical_state'])
    lowest_div.append(qc_stats[i]['lowest_values']['diversity'])
    lowest_real.append(qc_stats[i]['lowest_values']['realisticness'])
    lowest_spar.append(qc_stats[i]['lowest_values']['sparsity'])

range_val = [highest_val[i] - lowest_val[i] for i in range(len(highest_val))]
range_prox = [highest_prox[i] - lowest_prox[i] for i in range(len(highest_prox))]
range_crit = [highest_crit[i] - lowest_crit[i] for i in range(len(highest_crit))]
range_div = [highest_div[i] - lowest_div[i] for i in range(len(highest_div))]
range_real = [highest_real[i] - lowest_real[i] for i in range(len(highest_real))]
range_spar = [highest_spar[i] - lowest_spar[i] for i in range(len(highest_spar))]

relative_val = [chosen_val[i]/highest_val[i] for i in range(len(chosen_val))]
relative_prox = [chosen_prox[i]/highest_prox[i] for i in range(len(chosen_prox))]
relative_crit = [chosen_crit[i]/highest_crit[i] for i in range(len(chosen_crit))]
relative_div = [chosen_div[i]/highest_div[i] for i in range(1,len(chosen_div))]
relative_real = [chosen_real[i]/highest_real[i] for i in range(len(chosen_real))]
relative_spar = [chosen_spar[i]/highest_spar[i] for i in range(len(chosen_spar))]

# remove the first values of each correlation with diversity
qc_div_spears = qc_div_spears[1:]
val_div_spears = val_div_spears[1:]
prox_div_spears = prox_div_spears[1:]
crit_div_spears = crit_div_spears[1:]
qc_div_pears = qc_div_pears[1:]
val_div_pears = val_div_pears[1:]
prox_div_pears = prox_div_pears[1:]
crit_div_pears = crit_div_pears[1:] 

favorites_val = sum([1 for i in range(len(chosen_val)) if chosen_val[i] == highest_val[i]])/len(chosen_val)
favorites_prox = sum([1 for i in range(len(chosen_prox)) if chosen_prox[i] == highest_prox[i]])/len(chosen_prox)
favorites_crit = sum([1 for i in range(len(chosen_crit)) if chosen_crit[i] == highest_crit[i]])/len(chosen_crit)
favorites_div = sum([1 for i in range(len(chosen_div)) if chosen_div[i] == highest_div[i]])/len(chosen_div)
favorites_real = sum([1 for i in range(len(chosen_real)) if chosen_real[i] == highest_real[i]])/len(chosen_real)
favorites_spar = sum([1 for i in range(len(chosen_spar)) if chosen_spar[i] == highest_spar[i]])/len(chosen_spar)

print("qc-validity", round(np.mean(qc_val_spears),2), round(np.mean(qc_val_pears),2))
print("qc-proximity", round(np.mean(qc_prox_spears),2), round(np.mean(qc_prox_pears),2))
print("qc-critical_state", round(np.mean(qc_crit_spears),2), round(np.mean(qc_crit_pears),2))
print("qc-diversity", round(np.mean(qc_div_spears),2), round(np.mean(qc_div_pears),2))
print("qc-realisticness", round(np.mean(qc_real_spears),2), round(np.mean(qc_real_pears),2))
print("qc-sparsity", round(np.mean(qc_spar_spears),2), round(np.mean(qc_spar_pears),2))
print('')

print("validity-proximity", round(np.mean(val_prox_spears),2), round(np.mean(val_prox_pears),2))
print("validity-critical_state", round(np.mean(val_crit_spears),2), round(np.mean(val_crit_pears),2))
print("validity-diversity", round(np.mean(val_div_spears),2), round(np.mean(val_div_pears),2))
print("proximity-critical_state", round(np.mean(prox_crit_spears),2), round(np.mean(prox_crit_pears),2))
print("proximity-diversity", round(np.mean(prox_div_spears),2), round(np.mean(prox_div_pears),2))
print("critical_state-diversity", round(np.mean(crit_div_spears),2), round(np.mean(crit_div_pears),2))
print('')

print("pos_val", round(np.mean(pos_val),2))
print("pos_prox", round(np.mean(pos_prox),2))
print("pos_crit", round(np.mean(pos_crit),2))
print("pos_div", round(np.mean(pos_div),2))
print("pos_real", round(np.mean(pos_real),2))
print("pos_spar", round(np.mean(pos_spar),2))
print('')

print("chosen_val", round(np.mean(chosen_val),2))
print("chosen_prox", round(np.mean(chosen_prox),2))
print("chosen_crit", round(np.mean(chosen_crit),2))
print("chosen_div", round(np.mean(chosen_div),2))
print("chosen_real", round(np.mean(chosen_real),2))
print("chosen_spar", round(np.mean(chosen_spar),2))

table = [
    ['quality criteria', 'total value', 'percentile ranking', '%% favorites', 'qc-correlation', 'relative value', 'range', 'highest', 'lowest', 'contribution'],
    ['validity', round(np.mean(chosen_val),2), round(1-np.mean(pos_val),2), favorites_val, round(np.mean(qc_val_spears),2), round(np.mean(relative_val),2), round(np.mean(range_val),2), round(np.mean(highest_val),2), round(np.mean(lowest_val),2), round(np.mean(contribution_val),2)],
    ['proximity', round(np.mean(chosen_prox),2), round(1-np.mean(pos_prox),2), favorites_prox, round(np.mean(qc_prox_spears),2), round(np.mean(relative_prox),2), round(np.mean(range_prox),2), round(np.mean(highest_prox),2), round(np.mean(lowest_prox),2), round(np.mean(contribution_prox),2)],
    ['diversity', round(np.mean(chosen_div),2), round(1-np.mean(pos_div),2), favorites_div, round(np.mean(qc_div_spears),2), round(np.mean(relative_div),2), round(np.mean(range_div),2), round(np.mean(highest_div),2), round(np.mean(lowest_div),2), round(np.mean(contribution_div),2)],
    ['critical state', round(np.mean(chosen_crit),2), round(1-np.mean(pos_crit),2), favorites_crit, round(np.mean(qc_crit_spears),2), round(np.mean(relative_crit),2), round(np.mean(range_crit),2), round(np.mean(highest_crit),2), round(np.mean(lowest_crit),2), round(np.mean(contribution_crit),2)],
    ['realisticness', round(np.mean(chosen_real),2), round(1-np.mean(pos_real),2), favorites_real, round(np.mean(qc_real_spears),2), round(np.mean(relative_real),2), round(np.mean(range_real),2), round(np.mean(highest_real),2), round(np.mean(lowest_real),2), round(np.mean(contribution_real),2)],
    ['sparsity', round(np.mean(chosen_spar),2), round(1-np.mean(pos_spar),2), favorites_spar, round(np.mean(qc_spar_spears),2), round(np.mean(relative_spar),2), round(np.mean(range_spar),2), round(np.mean(highest_spar),2), round(np.mean(lowest_spar),2), round(np.mean(contribution_spar),2)],
]

print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

table = [
    ['quality criteria', 'percentile ranking', '\% favorites', 'relative value', 'qc-correlation'],
    ['', 'mean | median', 'mean', 'mean | median'],
    ['validity', str(round((1-np.mean(pos_val))*100,1)) + '\%' + " | " + str(round((1-np.median(pos_val))*100,1)) + '\%', str(round(favorites_val*100,2))+'\%', str(round(np.mean(relative_val),2))+" | "+str(round(np.median(relative_val),2)), str(round(np.mean(qc_val_spears),2))],
    ['proximity', str(round((1-np.mean(pos_prox))*100,1)) + '\%' + " | " + str(round((1-np.median(pos_prox))*100,1)) + '\%', str(round(favorites_prox*100,2))+'\%', str(round(np.mean(relative_prox),2))+" | "+str(round(np.median(relative_prox),2)), str(round(np.mean(qc_prox_spears),2))],
    ['diversity', str(round((1-np.mean(pos_div))*100,1)) + '\%' + " | " + str(round((1-np.median(pos_div))*100,1)) + '\%', str(round(favorites_div*100,2))+'\%', str(round(np.mean(relative_div),2))+" | "+str(round(np.median(relative_div),2)), str(round(np.mean(qc_div_spears),2))],
    ['critical state', str(round((1-np.mean(pos_crit))*100,1)) + '\%' + " | " + str(round((1-np.median(pos_crit))*100,1)) + '\%', str(round(favorites_crit*100,2))+'\%', str(round(np.mean(relative_crit),2))+" | "+str(round(np.median(relative_crit),2)), str(round(np.mean(qc_crit_spears),2))],
    ['realisticness', str(round((1-np.mean(pos_real))*100,1)) + '\%' + " | " + str(round((1-np.median(pos_real))*100,1)) + '\%', str(round(favorites_real*100,2))+'\%', str(round(np.mean(relative_real),2))+" | "+str(round(np.median(relative_real),2)), str(round(np.mean(qc_real_spears),2))],
    ['sparsity', str(round((1-np.mean(pos_spar))*100,1)) + '\%' + " | " + str(round((1-np.median(pos_spar))*100,1)) + '\%', str(round(favorites_spar*100,2))+'\%', str(round(np.mean(relative_spar),2))+" | "+str(round(np.median(relative_spar),2)), str(round(np.mean(qc_spar_spears),2))],
]

print(tabulate(table, headers='firstrow', tablefmt='latex_raw'))