from src.model import train_loop, Tree2Model
from src.tree import GPTree
from src.utils import concat

for i in range(1, 201):
    print(i)
    tree = GPTree()
    tree.gen_tree(10, init_symbol=concat)
    model = Tree2Model(tree.eval_tree())


train_loop(model, 2)