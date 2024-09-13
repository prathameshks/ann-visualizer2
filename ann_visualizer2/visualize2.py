from graphviz import Digraph
import keras
from keras.models import Sequential 
from keras.layers import Input,Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import json

def ann_viz(model:Sequential, view:bool=True, filename:str="network.gv", title:str="Neural Network")->None:
    """ Visualize a Keras Sequential model.

    # Arguments
        model: A Keras Sequential model instance.

        view: whether to display the model after generation.

        filename: where to save the visualization. (a .gv file)

        title: A title for the graph
    """
    
    input_layer_shape = 0
    hidden_layers_count = 0
    layer_types = []
    hidden_layers = []
    output_layer_shape = 0
    
    for layer in model.layers:  
        if(layer == model.layers[0]):
            # Check if model has input shape and assign to input_layer_shape
            try:
                input_layer_shape = model.input_shape[1]
            except ValueError:
                input_layer_shape = 0
            except:
                assert ("Invalid Model/ Model Not Supported Yet")

            hidden_layers_count += 1
            if (type(layer) == keras.layers.Dense):
                hidden_layers.append(int(layer.get_config()['units']))
                layer_types.append("Dense")
            else:
                hidden_layers.append(1)
                if (type(layer) == keras.layers.convolutional.Conv2D):
                    layer_types.append("Conv2D")
                elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                    layer_types.append("MaxPooling2D")
                elif (type(layer) == keras.layers.core.Dropout):
                    layer_types.append("Dropout")
                elif (type(layer) == keras.layers.core.Flatten):
                    layer_types.append("Flatten")
                elif (type(layer) == keras.layers.core.Activation):
                    layer_types.append("Activation")
        else:
            if(layer == model.layers[-1]):
                output_layer_shape = int(model.output_shape[1])
            else:
                hidden_layers_count += 1
                if (type(layer) == keras.layers.Dense):
                    hidden_layers.append(int(layer.get_config()['units']))
                    layer_types.append("Dense")
                else:
                    hidden_layers.append(1)
                    if (type(layer) == keras.layers.convolutional.Conv2D):
                        layer_types.append("Conv2D")
                    elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                        layer_types.append("MaxPooling2D")
                    elif (type(layer) == keras.layers.core.Dropout):
                        layer_types.append("Dropout")
                    elif (type(layer) == keras.layers.core.Flatten):
                        layer_types.append("Flatten")
                    elif (type(layer) == keras.layers.core.Activation):
                        layer_types.append("Activation")
                        
        last_layer_nodes = input_layer_shape
        nodes_up = input_layer_shape
        if(type(model.layers[0]) != keras.layers.Dense):
            last_layer_nodes = 1
            nodes_up = 1
            input_layer_shape = 1

        g = Digraph('g', filename=filename)
        n = 0
        g.graph_attr.update(splines="false", nodesep='1', ranksep='2')
        #Input Layer
        with g.subgraph(name='cluster_input') as c:
            if(type(model.layers[0]) == keras.layers.Dense):
                the_label = title+'\n\n\n\nInput Layer'
                if (int(model.layers[0].get_config()['units']) > 10):
                    the_label += " (+"+str(model.layers[0].get_config()['units'] - 10)+")"
                    input_layer_shape = 10
                c.attr(color='white')
                for i in range(0, input_layer_shape):
                    n += 1
                    c.node(str(n))
                    c.attr(label=the_label)
                    c.attr(rank='same')
                    c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

            elif(type(model.layers[0]) == keras.layers.convolutional.Conv2D):
                #Conv2D Input visualizing
                the_label = title+'\n\n\n\nInput Layer'
                c.attr(color="white", label=the_label)
                c.node_attr.update(shape="square")
                pxls = str(model.layers[0].input_shape).split(',')
                clr = int(pxls[3][1:-1])
                if (clr == 1):
                    clrmap = "Grayscale"
                    the_color = "black:white"
                elif (clr == 3):
                    clrmap = "RGB"
                    the_color = "#e74c3c:#3498db"
                else:
                    clrmap = ""
                c.node_attr.update(fontcolor="white", fillcolor=the_color, style="filled")
                n += 1
                c.node(str(n), label="Image\n"+pxls[1]+" x"+pxls[2]+" pixels\n"+clrmap, fontcolor="white")
            else:
                raise ValueError("ANN Visualizer: Layer not supported for visualizing")
        for i in range(0, hidden_layers_count):
            with g.subgraph(name="cluster_"+str(i+1)) as c:
                if (layer_types[i] == "Dense"):
                    c.attr(color='white')
                    c.attr(rank='same')
                    #If hidden_layers[i] > 10, dont include all
                    the_label = ""
                    if (int(str(model.layers[i].output_shape).split(",")[1][1:-1]) > 10):
                        the_label += " (+"+str(int(str(model.layers[i].output_shape).split(",")[1][1:-1]) - 10)+")"
                        hidden_layers[i] = 10
                    c.attr(labeljust="right", labelloc="b", label=the_label)
                    for j in range(0, hidden_layers[i]):
                        n += 1
                        c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db")
                        for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                            g.edge(str(h), str(n))
                    last_layer_nodes = hidden_layers[i]
                    nodes_up += hidden_layers[i]
                elif (layer_types[i] == "Conv2D"):
                    c.attr(style='filled', color='#5faad0')
                    n += 1
                    kernel_size = str(model.layers[i].get_config()['kernel_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['kernel_size']).split(',')[1][1 : -1]
                    filters = str(model.layers[i].get_config()['filters'])
                    c.node("conv_"+str(n), label="Convolutional Layer\nKernel Size: "+kernel_size+"\nFilters: "+filters, shape="square")
                    c.node(str(n), label=filters+"\nFeature Maps", shape="square")
                    g.edge("conv_"+str(n), str(n))
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), "conv_"+str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "MaxPooling2D"):
                    c.attr(color="white")
                    n += 1
                    pool_size = str(model.layers[i].get_config()['pool_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['pool_size']).split(',')[1][1 : -1]
                    c.node(str(n), label="Max Pooling\nPool Size: "+pool_size, style="filled", fillcolor="#8e44ad", fontcolor="white")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "Flatten"):
                    n += 1
                    c.attr(color="white")
                    c.node(str(n), label="Flattening", shape="invtriangle", style="filled", fillcolor="#2c3e50", fontcolor="white")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "Dropout"):
                    n += 1
                    c.attr(color="white")
                    c.node(str(n), label="Dropout Layer", style="filled", fontcolor="white", fillcolor="#f39c12")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "Activation"):
                    n += 1
                    c.attr(color="white")
                    fnc = model.layers[i].get_config()['activation']
                    c.node(str(n), shape="octagon", label="Activation Layer\nFunction: "+fnc, style="filled", fontcolor="white", fillcolor="#00b894")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1


        with g.subgraph(name='cluster_output') as c:
            if (type(model.layers[-1]) == keras.layers.Dense):
                c.attr(color='white')
                c.attr(rank='same')
                c.attr(labeljust="1")
                for i in range(1, output_layer_shape+1):
                    n += 1
                    c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                c.attr(label='Output Layer', labelloc="bottom")
                c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

        g.attr(arrowShape="none")
        g.edge_attr.update(arrowhead="none", color="#707070")
        if view == True:
            g.view()
