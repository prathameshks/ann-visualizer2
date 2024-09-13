import graphviz
from tensorflow import keras

def ann_viz(model, view=True, filename="improved_network", title="Neural Network"):
    """
    Visualize a Keras Sequential model.
    
    Args:
    model (keras.Model): A Keras model instance.
    view (bool): Whether to display the model after generation.
    filename (str): Where to save the visualization (without extension).
    title (str): A title for the graph.
    """
    
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR', size='12,8')
    
    dot.attr('node', shape='circle', style='filled', color='lightblue')
    
    # Input Layer
    with dot.subgraph(name='cluster_input') as c:
        c.attr(color='lightgrey', label='Input Layer')
        try:
            input_shape = model.input_shape
        except:
            input_shape = 1
        if isinstance(input_shape, tuple):
            input_nodes = input_shape[1] if len(input_shape) > 1 else 1
        elif isinstance(input_shape, list):
            input_nodes = input_shape[0][1] if len(input_shape[0]) > 1 else 1
        else:
            input_nodes = 1
        for i in range(input_nodes):
            c.node(f'i{i}')
    
    # Hidden Layers and Output Layer
    for i, layer in enumerate(model.layers):
        layer_config = layer.get_config()
        layer_type = layer.__class__.__name__
        with dot.subgraph(name=f'cluster_{i}') as c:
            if i == len(model.layers) - 1:
                c.attr(color='lightgrey', label='Output Layer')
                color = 'lightgreen'
            else:
                c.attr(color='lightgrey', label=f'Hidden Layer {i+1}')
                color = 'lightyellow'
            
            c.attr('node', style='filled', color=color)
            
            if isinstance(layer, keras.layers.Dense):
                units = layer_config['units']
                activation = layer_config['activation']
                for j in range(units):
                    c.node(f'h{i}_{j}', f'{activation}')
            elif isinstance(layer, keras.layers.Dropout):
                rate = layer_config['rate']
                c.node(f'dropout_{i}', f'Dropout\n{rate}')
            else:
                c.node(f'layer_{i}', layer_type)
    
    # Connections
    for i in range(len(model.layers)):
        if i == 0:
            for j in range(input_nodes):
                if isinstance(model.layers[0], keras.layers.Dense):
                    units = model.layers[0].get_config()['units']
                    for k in range(units):
                        dot.edge(f'i{j}', f'h0_{k}')
                else:
                    dot.edge(f'i{j}', f'layer_0')
        else:
            prev_layer = model.layers[i-1]
            curr_layer = model.layers[i]
            prev_units = prev_layer.get_config().get('units', 1)
            curr_units = curr_layer.get_config().get('units', 1)
            
            for j in range(prev_units):
                for k in range(curr_units):
                    dot.edge(f'h{i-1}_{j}' if isinstance(prev_layer, keras.layers.Dense) else f'layer_{i-1}',
                             f'h{i}_{k}' if isinstance(curr_layer, keras.layers.Dense) else f'layer_{i}')
    
    # Add title
    dot.attr(label=title, labelloc='t', fontsize='20')
    
    # Render the graph
    dot.render(filename, view=view, format='pdf', cleanup=True)
    print(f"Visualization saved as {filename}.pdf")

# Example usage
if __name__ == "__main__":
    model = keras.Sequential([
        keras.layers.Dense(6, activation='relu', input_shape=(11,)),
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    ann_viz(model, title="Sample Neural Network")