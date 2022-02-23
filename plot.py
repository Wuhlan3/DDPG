import matplotlib.pyplot as plt
episode_list = []
reward_list = []
with open("LunarLanderContinuous-v2.txt", "r") as f: 
    line = f.readlines() 
for i in line:
    l = i.split()
    #print(l[1],l[2])
    episode_list.append(int(l[1]))
    reward_list.append(float(l[2]))

plt.plot(episode_list,reward_list,"r",label="LunarLanderContinuous-v2")
plt.ylim(-1300,200)
plt.xlabel("Steps")
plt.ylabel("Rewards")
plt.legend()
plt.show()