"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network.
    If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""


from models.avsg_model import AvsgModel
from models.base_model import BaseModel


#---------------------------------------------------------------------

def get_model_class(model_name):
    if model_name == 'avsg':
        model_class = AvsgModel
    else:
        raise NotImplementedError
    return model_class

#---------------------------------------------------------------------
def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = get_model_class(model_name)
    return model_class.modify_commandline_options

#---------------------------------------------------------------------

def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    """
    model_class = get_model_class(opt.model)
    instance = model_class(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
