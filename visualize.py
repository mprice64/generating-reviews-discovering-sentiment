# Launch webserver with
#   python visualize.py
# and go to http://127.0.0.1:5000

import encoder
import numpy as np
from flask import Flask, render_template, request

notes = """
2388 -- Sentiment neuron
1742 2045 2094 -- Position in review?
2903 -- Sentiment also ?
3353 -- Sentence position
"""

mdl = encoder.Model()
SENTIMENT_NEURON = 2388

# Adapted from https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    r = int(max(0, 255*(1 - ratio)))
    g = int(max(0, 255*(ratio - 1)))
    b = 255 - g - r
    return (r, g, b)

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('viz-text.html',
                           layers=str(SENTIMENT_NEURON)
                           )

@app.route("/features")
def features():
    text = request.args.get("text","")
    layers = request.args.get("layers",str(SENTIMENT_NEURON))

    # TODO -- Is there a more optimal way to get the activations at each letter?  Seems like it should be fetchable
    # from the iterative processing somehow.
    # Perhaps that is what https://github.com/racoder/generating-reviews-discovering-sentiment/commit/86beb6bbe1181b1fc102c4afc69daa34c82df171 is ?

    # This generates each substring "F", "Fo", "Foo", etc.
    texts = [ text[0:i] for i in range(1,len(text)+1) ]

    text_features = mdl.transform(texts)

    all_paired_text = []

    if len(layers):
        layers_array = list(map(lambda l: int(l), layers.split(",")))
    else:
        layers_array = list(range(0,len(text_features[0])))

    text_features = text_features[:, layers_array] # = text_features[:, SENTIMENT_NEURON]

    # exampleColors = np.linspace(-1,1,len(text_features[0]))
    # text_features = np.vstack([exampleColors, text_features])

    for layerN, activations in zip( layers_array, text_features.T):
        # activations = np.linspace(-1,1, len(activations))

        min = np.min( activations )
        max = np.max( activations )

        # print( "Min: {:.2} Max: {:.2}".format( float(min),float(max) ) )

        colors = map( lambda n: rgb(min,max,float(n)), activations)

        colors_wrapped = map( lambda c: "rgb({},{},{})".format(*c), colors)
        paired_text = list(zip( text, colors_wrapped ))

        splits = np.array_split( np.array(paired_text), np.ceil(len(paired_text)/150))
        for split in splits:
            # split = np.hstack([[str(layerN),"rgb(0,0,0)"],split])
            # split.insert(0,[str(layerN),"rgb(0,0,0)"])
            rowLabel = [str(layerN),"rgb(0,0,0)"]
            # split = rowLabel + split
            split = np.insert( split, 0, np.array(rowLabel), axis=0)
            all_paired_text.append(split)

        # all_paired_text.append( paired_text )

    return render_template('viz-text.html', text=text, layers=layers, sentiments=all_paired_text)

if __name__ == "__main__":
    app.run()
