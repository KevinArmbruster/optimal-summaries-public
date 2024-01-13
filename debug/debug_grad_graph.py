# %%
from graphviz import Digraph
from graphviz import Source
import torch
from torch.autograd import Variable, Function


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return True
        grad_output = grad_output.data
        return torch.isnan(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                
                sizes = [str(gi.shape) + f" ({gi.numel()})" if gi != None else "None" for gi in fn_dict[fn]]
                
                node_name = str(type(fn).__name__) + "\n " + ", ".join(sizes)
                dot.node(str(id(fn)), node_name, fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

if __name__ == '__main__':
    x = Variable(torch.randn(10, 10), requires_grad=True)
    y = Variable(torch.randn(10, 10), requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    get_dot = register_hooks(z)
    z.backward()
    dot = get_dot()
    path = 'aDebugGraph.dot'
    dot.save(path)
    print(dot)
    
    s = Source.from_file(path)
    s.render('aDebugGraph', format='png', cleanup=True)



