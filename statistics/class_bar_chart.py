import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


with open ('data/new_merged.json',"r") as f:
    data = json.load(f)


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
class_labels = ['Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup', 'Lid', 'Other', 'Plastic Bag', 'Pop tab', 'Straw']
class_count = {}
for i in range(10):
    class_count[i+1] = 0


for annotation in data['annotations']:
    id = annotation.get('category_id')

    class_count[id] += 1



plt.style.use('ggplot')

my_cmap = cm.get_cmap('jet')


x_pos = [i for i, _ in enumerate(class_labels)]

plt.bar(x_pos, class_count.values(), color=colors)
plt.xlabel("Category", fontsize=16)
plt.ylabel("Number of annotations", fontsize=16)


plt.xticks(x_pos, class_labels, rotation=45)

plt.show()

    
    