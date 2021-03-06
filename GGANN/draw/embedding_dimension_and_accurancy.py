import matplotlib.pyplot as plt


# 不同节点嵌入维度下的模型分类精度
def avg(arr):
    res = 0
    for i in arr:
        res += i
    return res/(len(arr))


def show_avg(arr1, title1, arr2, title2, arr3, title3, arr4, title4):
    print(title1+": "+str(avg(arr1)))
    print(title2+": "+str(avg(arr2)))
    print(title3+": "+str(avg(arr3)))
    print(title4+": "+str(avg(arr4)))

test_190_acc = \
    [0.98619, 0.98469, 0.98531, 0.98575, 0.97712, 0.96727, 0.98636, 0.98504, 0.97061, 0.98935, 0.98583,
     0.97695, 0.98196, 0.97519, 0.96322, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627, 0.98496, 0.98020,
     0.98654, 0.97633, 0.98152, 0.98601, 0.96481, 0.95099, 0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
     0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742, 0.98566, 0.98575, 0.98540, 0.97413, 0.98152]
test_220_acc = \
    [0.98619, 0.98469, 0.98531, 0.98575, 0.97712, 0.96727, 0.98636, 0.98504, 0.97061, 0.98935, 0.98583,
     0.97695, 0.98196, 0.97519, 0.96322, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627, 0.98496, 0.98020,
     0.98654, 0.97633, 0.98152, 0.98601, 0.96481, 0.95099, 0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
     0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742, 0.98566, 0.98575, 0.98540, 0.97413, 0.98152]
test_250_acc = \
    [0.98619, 0.98469, 0.98531, 0.98575, 0.97712, 0.96727, 0.98636, 0.98504, 0.97061, 0.98935, 0.98583,
     0.97695, 0.98196, 0.97519, 0.96322, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627, 0.98496, 0.98020,
     0.98654, 0.97633, 0.98152, 0.98601, 0.96481, 0.95099, 0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
     0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742, 0.98566, 0.98575, 0.98540, 0.97413, 0.98152]

test_270_acc = \
    [0.98619, 0.98469, 0.98531, 0.98575, 0.97712, 0.96727, 0.98636, 0.98504, 0.97061, 0.98935, 0.98583,
     0.97695, 0.98196, 0.97519, 0.96322, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627, 0.98496, 0.98020,
     0.98654, 0.97633, 0.98152, 0.98601, 0.96481, 0.95099, 0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
     0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742, 0.98566, 0.98575, 0.98540, 0.97413, 0.98152]

test_300_acc = \
    [0.98619, 0.98469, 0.98531, 0.98575, 0.97712, 0.96727, 0.98636, 0.98504, 0.97061, 0.98935, 0.98583,
     0.97695, 0.98196, 0.97519, 0.96322, 0.98733, 0.97941, 0.94343, 0.98399, 0.98627, 0.98496, 0.98020,
     0.98654, 0.97633, 0.98152, 0.98601, 0.96481, 0.95099, 0.98232, 0.98188, 0.96841, 0.96058, 0.98020,
     0.97624, 0.98487, 0.98452, 0.97457, 0.98733, 0.95654, 0.95742, 0.98566, 0.98575, 0.98540, 0.97413, 0.98152]

valid_instance_sec = [5.21, 4.81, 4.52, 4.09, 3.67, 2.78, 2.56]
test_instance_sec = [15.95, 14.97, 14.45, 12.34, 11.19, 8.58, 6.78]
valid_loss = [0.5, 0.5, 0.5, 0.5, 0.48159, 0.5, 0.50008]
test_loss = [0.50002, 0.50014, 0.50013, 0.50006, 0.48329, 0.5, 0.50021]

x = [190, 200, 220, 250, 270, 300, 350]

figure = plt.figure(figsize=(10, 10), dpi=80)

plt.subplot(221)
plt.plot(x, valid_instance_sec, "b:", label='Training period', linewidth=2, marker='o')
plt.plot(x, test_instance_sec, "r:", label='Test period', linewidth=2, marker='^')
# plt.plot(x, test_250_acc, "c:", label='valid_AST_GGNN', linewidth=2, marker="v")
# plt.plot(x, test_270_acc, "m:", label='valid_AST_TBCNN', linewidth=2, marker="*")


plt.xlabel("Embeddings Dimension")
plt.ylabel("Processing graph per second")
plt.ylim(0, 17)
plt.xlim(180, 360)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

plt.subplot(222)
plt.plot(x, valid_loss, "b:", label="Training period", linewidth=2, marker='o')
plt.plot(x, test_loss, "r:", label="Test period", linewidth=2, marker='^')


plt.xlabel("Embeddings Dimension")
plt.ylabel("Loss")
plt.ylim(0.45, 0.55)
plt.xlim(180, 360)
plt.legend()
plt.grid(True, linestyle="--", color="#C9C9C9", linewidth="1")

plt.show()
plt.savefig("dimension")


