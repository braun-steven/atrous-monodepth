import matplotlib.pyplot as plt


aspp_max = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
abs_rels=[0.1035, 0.1058, 0.1074, 0.1075, 0.1075, 0.1073, 0.1077, 0.1075, 0.1086, 0.1074]

plt.figure()
plt.title("Influence of ASPP Modules")
plt.xlabel("ASPP Max Module Size")
plt.ylabel("Test set abs_rel")
plt.plot(aspp_max, abs_rels)
plt.xticks(aspp_max)
plt.savefig("aspp_plot.png")

