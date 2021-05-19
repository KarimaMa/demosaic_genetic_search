import sys, torch, numpy, os, inspect, base64

from imageio import imread, imsave

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import demosaic_ast, torch_model, cost

def linearize_ast(node, emitted):
    if id(node) in emitted:
        return []

    ops = []
    if hasattr(node, 'child'):
        ops = linearize_ast(node.child, emitted)
        l = len(emitted)
        emitted[id(node)] = l
        ops.append((l, node, emitted[id(node.child)], -1, -1))
    elif hasattr(node, 'child1'):
        assert(hasattr(node, 'child2'))
        assert(hasattr(node, 'child3'))
        ops = linearize_ast(node.child1, emitted)
        ops += linearize_ast(node.child2, emitted)
        ops += linearize_ast(node.child3, emitted)        
        l = len(emitted)
        emitted[id(node)] = l
        ops.append((l, node, emitted[id(node.child1)], emitted[id(node.child2)], emitted[id(node.child3)]))
    elif hasattr(node, 'lchild'):
        assert(hasattr(node, 'rchild'))
        ops = linearize_ast(node.lchild, emitted)
        ops += linearize_ast(node.rchild, emitted)
        l = len(emitted)
        emitted[id(node)] = l
        ops.append((l, node, emitted[id(node.lchild)], emitted[id(node.rchild)], -1))
    elif hasattr(node, 'node'):
        ops = linearize_ast(node.node, emitted)
        l = len(emitted)
        emitted[id(node)] = l
        ops.append((l, node, emitted[id(node.node)], -1, -1))        
    else:
        l = len(emitted)
        emitted[id(node)] = l
        ops.append((l, node, -1, -1, -1))
        
    return ops
        
if __name__ == '__main__':

    assert(len(sys.argv) == 5)
    green_asts = {}
    idx = 0
    for filename in open(sys.argv[1]):           
        green_ast = demosaic_ast.load_ast(filename.strip())        
        green_asts[idx] = green_ast
        idx += 1
        
    ast = demosaic_ast.load_ast(sys.argv[2])

    # Graft in the green model
    def graft_green_model(ast):        
        for n in ast.preorder():
            if type(n) is demosaic_ast.Input and n.name == "Input(GreenExtractor)" and hasattr(n, 'green_model_id'):
                n.node = green_asts[n.green_model_id]
                print(n.dump())
                print(n.node.dump())
                del n.green_model_id
            elif hasattr(n, 'node'):
                graft_green_model(n.node)

    graft_green_model(ast)
                
    print(ast.dump())

    # Print the cost
    print("COST: ", cost.ModelEvaluator(None).compute_cost(ast))
    
    model = ast.ast_to_model()
    try:
        model.load_state_dict(torch.load(sys.argv[3], map_location = torch.device("cpu")))
    except:
        print("Failed to load weights. Oh well.")

    # Run the model on a test pattern to give us a unit test
    """
    circle = numpy.array(imread("circle.png")).astype(numpy.float32) / (2**8 - 1)
    bayer = numpy.zeros((1, 4, 64, 64))
    bayer[0,0,:,:] = circle[0::2, 0::2]
    bayer[0,1,:,:] = circle[0::2, 1::2]
    bayer[0,2,:,:] = circle[1::2, 0::2]
    bayer[0,3,:,:] = circle[1::2, 1::2]    
    bayer = torch.Tensor(bayer)
    rb_bayer = bayer[:,1:3,:,:]
    g_bayer = bayer[:,0::3,:,:]    
    model_inputs = {'Input(Mosaic)' : bayer,
                    'Input(RedBlueBayer)' : rb_bayer,
                    'Input(Green@GrGb)': g_bayer}
    output = model.run(model_inputs)
    output = output.detach().numpy()[0, :, :, :]
    output *= 255
    output = numpy.clip(output, 0, 255)
    output = output.astype(numpy.uint8)
    output = output.transpose(1, 2, 0)
    imsave(os.path.join(os.path.dirname(sys.argv[4]), "circle_out_ref.png"), output)
    """
    
    # Assign weights to the AST nodes
    def assign_weights(node):
        for n in node.preorder():
            n.weights = [None, None]
        
            if hasattr(n, 'model'):
                if hasattr(n.model, 'param_name'):
                    f = getattr(n.model, n.model.param_name)
                    if hasattr(f, 'weight'):
                        n.weights = [f.weight.detach().numpy(), None]
                elif hasattr(n.model, 'param_name_v'):
                    f = getattr(n.model, n.model.param_name_v)
                    n.weights = [f.weight.detach().numpy()]
                    f = getattr(n.model, n.model.param_name_h)
                    n.weights.append(f.weight.detach().numpy())                

            if type(n) is demosaic_ast.Input and hasattr(n, 'node'):
                assign_weights(n.node)

    assign_weights(ast)
    
    # Linearize the AST to SSA form
    with open(sys.argv[4], 'w') as f:
        ops = linearize_ast(ast, {})
        for (i, n, a, b, c) in ops:

            if (n.name == "BilinearUpsample"):
                print(dir(n))
                
            if hasattr(n, 'out_c'):
                out_c = n.out_c
            else:
                out_c = 0

            if hasattr(n, 'in_c'):
                in_c = n.in_c
            else:
                in_c = 0

            if type(in_c) is tuple:
                in_c = list(in_c)
            else:
                in_c = [in_c]
            while len(in_c) < 3:
                in_c.append(0)
            in_c = ' '.join(map(str, in_c))

            if hasattr(n, 'groups'):
                groups = n.groups
            else:
                groups = 0

            if hasattr(n, 'factor'):
                factor = int(n.factor)
            else:
                factor = 0
                
            weights0 = "0 0 0 0 None"
            weights1 = "0 0 0 0 None"
            if n.weights[0] is not None:
                n.weights[0] = n.weights[0].astype(numpy.float32)
                weights0 = ' '.join(map(str, n.weights[0].shape))
                weights0 += ' ' + base64.b64encode(n.weights[0]).decode('utf-8')
                
            if n.weights[1] is not None:
                n.weights[1] = n.weights[1].astype(numpy.float32)
                weights1 = ' '.join(map(str, n.weights[1].shape))
                weights1 += ' ' + base64.b64encode(n.weights[1]).decode('utf-8')

            print(i, n.name, out_c, in_c, groups, a, b, c, factor, weights0, weights1, file=f)



            
