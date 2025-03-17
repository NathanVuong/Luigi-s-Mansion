import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#graph all runs
# runs = [(run_name, x pos)]

img = mpimg.imread(r'C:\Users\19083\Desktop\175\level_layout.png')
fig, ax = plt.subplots()
ax.imshow(img)

height, width, _ = img.shape
for x in range(0, width, 100):
    ax.axvline(x=x, color='red', linestyle='--', linewidth=1)
# for y in range(0, height, 100):
#     ax.axhline(y=y, color='red', linestyle='--', linewidth=1)

ax.set_xticks(range(0, width, 100))
ax.set_yticks(range(0, height, 100))
x_labels = [str(i // 100 + 1) for i in range(0, width, 100)]
y_labels = [str(i // 100 + 1) for i in range(0, height, 100)]
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)

plt.savefig('level_layout_marked.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.tight_layout()