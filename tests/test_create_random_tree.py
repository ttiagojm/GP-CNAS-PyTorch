from src.tree import GPTree
from src.model import Tree2Model, train_loop
from src.utils import concat

# Generate various random trees

tree = GPTree()
tree.gen_tree(10, init_symbol=concat)
layers = tree.eval_tree()

print(tree)

model = Tree2Model(layers)

train_loop(model, 2)

# for i in range(10):
#     tree = GPTree()
#     tree.gen_tree(10, is_grow=i % 2 == 0, init_symbol=concat)
#     print(tree, end="\n##################\n")
