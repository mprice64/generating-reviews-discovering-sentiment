# Launch webserver with
#   python visualize.py
# and go to http://127.0.0.1:5000

import encoder
import numpy as np
from flask import Flask, render_template, request

# TODO -- Move this somewhere else.. like a jupyter notebook.
NOTES = """
2388 -- Sentiment neuron
1742 2045 3665 -- Position in review?
2094 -- First letter, sometimes ? 
2903 -- Sentiment also ?
3353 -- Sentence position

1668 - The "the" predictor?
2428 - Quote/Paren nested level ?
2846 - Quoted block 
2868 - parens 
2882 ? 
3090 -- Donno, rolls interestingly
3401 -- Number reactive..
3555, 3471 -- Slow moving
3781 - Another quote/parent thing ?
3929 slow moving 
46 - ? 
"""

INTERESTING_NEURONS = [2388,1742,2045,2094,3665,2903,3353,1668,2428,2846,2868,2882,3090,3401,3555,3471,3781,3929,46]

mdl = encoder.Model()

SENTIMENT_NEURON = 2388

EXAMPLE_REVIEW = """Team Spirit is maybe made by the best intentions, but it misses the warmth
of "All Stars" (1997) by Jean van de Velde. Most scenes are identic, just
not that funny and not that well done. The actors repeat the same lines as
in "All Stars" but without much feeling."""

# Adapted from https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
# TODO -- Change this to more closely match the paper's color mapping ?
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
                           layers=str(SENTIMENT_NEURON),
                           text=EXAMPLE_REVIEW
                           )

@app.route("/features")
def features():

    text = request.args.get("text",EXAMPLE_REVIEW)
    layers = request.args.get("layers",str(SENTIMENT_NEURON))
    max_row_width = int( request.args.get("max_row_width", "150") )

    # TODO -- Make method configurable?  This is the old method, probably will just remove.
    # keeping it around for now incase a comparison needs to be made.
    if False:
        # This generates each substring "F", "Fo", "Foo", etc.
        texts = [ text[0:i] for i in range(1,len(text)+1) ]
        text_features = mdl.transform(texts)
    else:
        text_features = mdl.transform([text], True)

    all_paired_text = []

    if len(layers):
        layers_array = list(map(lambda l: int(l), layers.split(",")))
    else:
        layers_array = list(range(0,len(text_features[0])))

    text_features = text_features[:, layers_array]

    # exampleColors = np.linspace(-1,1,len(text_features[0]))
    # text_features = np.vstack([exampleColors, text_features])

    for layerN, activations in zip( layers_array, text_features.T):
        # This is an issue, where the min/max of the color range is per-neuron and per-example.
        # This means when you feed in different inputs, the colors will have different meanings.
        # But if you use a global min/max across all neurons, many neurons get totally washed out.

        min = np.min( activations )
        max = np.max( activations )

        # print( "Min: {:.2} Max: {:.2}".format( float(min),float(max) ) )

        colors = map(lambda n: rgb(min,max,float(n)), activations)

        colors_wrapped = map(lambda c: "rgb({},{},{})".format(*c), colors)
        paired_text = list(zip(encoder.preprocess_text(text), colors_wrapped ))

        splits = np.array_split(np.array(paired_text), np.ceil(len(paired_text)/ max_row_width))
        for split in splits:
            # split = np.hstack([[str(layerN),"rgb(0,0,0)"],split])
            # split.insert(0,[str(layerN),"rgb(0,0,0)"])
            rowLabel = [str(layerN),"rgb(0,0,0)"]
            # split = rowLabel + split
            split = np.insert( split, 0, np.array(rowLabel), axis=0)
            all_paired_text.append(split)

    return render_template('viz-text.html', text=text, layers=layers, sentiments=all_paired_text)

if __name__ == "__main__":
    app.run()
