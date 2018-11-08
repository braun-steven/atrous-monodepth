from .backbone import Backbone

def get_model(model, input_channels=3):
    if model == 'backbone':
        out_model = Backbone(input_channels=input_channels)
    # elif and so on and so on
    else:
        raise ValueError("Please specify a valid model")
    return out_model
