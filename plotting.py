from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

'''
#experiment 1b
y1 = [12.16, 09.73, 11.82, 10.88, 10.98]
y2 = [40.53, 46.44, 42.65, 47.78, 44.62]
y4 = [90.07, 91.66, 91.30, 91.11, 97.61]
y8 = [98.7, 99.7, 91.6, 98.3, 97.5]
y16 = [98.38, 99.47, 99.6, 99.74, 97.39]
y32 = [98.7, 96.6, 99.4, 99.1, 99]
y64 = [99.6, 99.9, 99, 99.8, 99.4]

y1_mean = np.mean(y1)
y2_mean = np.mean(y2)
y4_mean = np.mean(y4)
y8_mean = np.mean(y8)
y16_mean = np.mean(y16)
y32_mean = np.mean(y32)
y64_mean = np.mean(y64)

y1_std = np.std(y1)
y2_std = np.std(y2)
y4_std = np.std(y4)
y8_std = np.std(y8)
y16_std = np.std(y16)
y32_std = np.std(y32)
y64_std = np.std(y64)

means = [y1_mean, y2_mean, y4_mean, y8_mean, y16_mean, y32_mean, y64_mean]
error = [y1_std, y2_std, y4_std, y8_std, y16_std, y32_std, y64_std]


x = np.arange(len(means))
x_pos = ["1 %", "2 %", "4 %","8 %","16 %","32 %","64 %"]

fig, ax = plt.subplots()
ax.bar(x, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Accuracy on new commands (%)')
ax.set_xticks(x)
ax.set_ylim(top=100)
ax.set_xticklabels(x_pos)
ax.set_xlabel('Percent of commands used for training')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('1b.png')
plt.show()
'''

'''
#experiment 2, action sequence length
g24 = [0.9494047619047619, 0.8303571428571428, 0.7708333333333334, 0.9583333333333334, 0.8839285714285714]
g25 = [0.5714285714285714, 0.4419642857142857, 0.5334821428571428, 0.5691964285714286, 0.5714285714285714]
g26 = [0.0, 0.0, 0.0, 0.0, 0.0]
g27 = [0.0, 0.0, 0.0, 0.0, 0.0]
g28 = [0.046875, 0.004464285714285698, 0.0, 0.022321428571428603, 0.0]
g30 = [0.0, 0.0, 0.0, 0.0, 0.0]
g32 = [0.0, 0.004464285714285698, 0.0, 0.024553571428571397, 0.0]
g33 = [0.0, 0.0, 0.0, 0.0, 0.0]
g36 = [0.0, 0.0, 0.0, 0.0, 0.0]
g40 = [0.0, 0.0, 0.0, 0.0, 0.0]
g48 = [0.0, 0.0, 0.0, 0.0, 0.0]

g24 = [x*100 for x in g24]
g25 = [x*100 for x in g25]
g26 = [x*100 for x in g26]
g27 = [x*100 for x in g27]
g28 = [x*100 for x in g28]
g30 = [x*100 for x in g30]
g32 = [x*100 for x in g32]
g33 = [x*100 for x in g33]
g36 = [x*100 for x in g36]
g40 = [x*100 for x in g40]
g48 = [x*100 for x in g48]

g24_mean = np.mean(g24)
g25_mean = np.mean(g25)
g26_mean = np.mean(g26)
g27_mean = np.mean(g27)
g28_mean = np.mean(g28)
g30_mean = np.mean(g30)
g32_mean = np.mean(g32)
g33_mean = np.mean(g33)
g36_mean = np.mean(g36)
g40_mean = np.mean(g40)
g48_mean = np.mean(g48)

g24_std = np.std(g24)
g25_std = np.std(g25)
g26_std = np.std(g26)
g27_std = np.std(g27)
g28_std = np.std(g28)
g30_std = np.std(g30)
g32_std = np.std(g32)
g33_std = np.std(g33)
g36_std = np.std(g36)
g40_std = np.std(g40)
g48_std = np.std(g48)

means = [g24_mean, g25_mean, g26_mean, g27_mean, g28_mean, g30_mean, g32_mean, g33_mean, g36_mean, g40_mean, g48_mean]
error = [g24_std, g25_std, g26_std, g27_std, g28_std, g30_std, g32_std, g33_std, g36_std, g40_std, g48_std]


x = np.arange(len(means))
x_pos = ["24", "25", "26", "27", "28", "30", "32", "33", "36", "40", "48"]

fig, ax = plt.subplots()
ax.bar(x, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylim(top=100)
ax.set_ylabel('Accuracy on new commands (%)')
ax.set_xticks(x)
ax.set_xticklabels(x_pos)
ax.set_xlabel('Ground-truth action sequence length')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('2_action.png')
plt.show()
'''

'''
#experiment 2, command sequence length
c4 = [0.0, 0.0, 0.0, 0.0, 0.0]
c6 = [0.0, 0.0, 0.0, 0.0, 0.0]
c7 = [0.0, 0.0, 0.0, 0.0, 0.0]
c8 = [0.16666666666666663, 0.15755208333333337, 0.14453125, 0.16666666666666663, 0.16145833333333337]
c9 = [0.20334928229665072, 0.14294258373205737, 0.16507177033492826, 0.20454545454545459, 0.1824162679425837]

c4 = [x*100 for x in c4]
c6 = [x*100 for x in c6]
c7 = [x*100 for x in c7]
c8 = [x*100 for x in c8]
c9 = [x*100 for x in c9]


c4_mean = np.mean(c4)
c6_mean = np.mean(c6)
c7_mean = np.mean(c7)
c8_mean = np.mean(c8)
c9_mean = np.mean(c9)

c4_std = np.std(c4)
c6_std = np.std(c6)
c7_std = np.std(c7)
c8_std = np.std(c8)
c9_std = np.std(c9)

means = [c4_mean, c6_mean, c7_mean, c8_mean, c9_mean]
error = [c4_std, c6_std, c7_std, c8_std, c9_std]


x = np.arange(len(means))
x_pos = ["4", "6", "7", "8", "9"]

fig, ax = plt.subplots()
ax.bar(x, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylim(top=100)
ax.set_ylabel('Accuracy on new commands (%)')
ax.set_xticks(x)
ax.set_xticklabels(x_pos)
ax.set_xlabel('Command length')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('2_command.png')
plt.show()
'''
'''
# experiment 3, composite commands
y1 = [0.0, 0.0, 0.0, 0.0, 0.0]
y2 = [1.0, 8.2, 0.6, 0.2, 1.0]
y4 = [4.2, 1.4, 4.4, 2.3, 2.4]
y8 = [13, 3, 15.5, 6, 16.1]
y16 = [33.25, 69.67, 48.2, 62.7, 31]
y32 = [79.9,62.8,80.6,82.7,77.6]

y1_mean = np.mean(y1)
y2_mean = np.mean(y2)
y4_mean = np.mean(y4)
y8_mean = np.mean(y8)
y16_mean = np.mean(y16)
y32_mean = np.mean(y32)

y1_std = np.std(y1)
y2_std = np.std(y2)
y4_std = np.std(y4)
y8_std = np.std(y8)
y16_std = np.std(y16)
y32_std = np.std(y32)

means = [y1_mean, y2_mean, y4_mean, y8_mean, y16_mean, y32_mean]
error = [y1_std, y2_std, y4_std, y8_std, y16_std, y32_std]


x = np.arange(len(means))
x_pos = ["1", "2", "4", "8", "16", "32"]

fig, ax = plt.subplots()
ax.bar(x, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Accuracy on new commands (%)')
ax.set_ylim(top=100)
ax.set_xticks(x)
ax.set_xticklabels(x_pos)
ax.set_xlabel('Number of composed commands used for training')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('comp_commands.png')
plt.show()
'''