
#广度
fig_1 = plt.figure(figsize=(7,7))
ax_1 =fig_1.add_subplot("test1")

for i in np.arrange(len(key_list)):
    for j in ['error(train)','error(valid)']:
        ax_1.plot(np.arrange(1,stats.shape[0])* stats_interval,
                         stats[1:,keys_list[i][j]],label='width'+j+str(32*2**i))
                  ax.
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('Accuracy')


#深度
fig_1 = plt.figure(figsize=(7, 7))
ax_1 = fig_1.add_subplot("test1")

for i in np.arrange(len(key_list)):
        for j in ['error(train)', 'error(valid)']:
          ax_1.plot(np.arrange(1, stats.shape[0]) * stats_interval,
                  stats[1:, keys_list[i][j]], label='depth' + j + str(32 * 2 ** i))
           ax.
            ax_1.legend(loc=0)
            ax_1.set_xlabel('Epoch number')
            ax_1.set_ylabel('Accuracy')


#dropout，模型










#dropput,测试








#L1模型















#L1测试














#L2模型






















#L2测试