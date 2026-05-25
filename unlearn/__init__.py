from .gradient_ascent import GradAscent
from .scrub import SCRUB
from .salun import SalUn
from .unlearn_method import UnlearnMethod

def get_unlearn_method(name) -> UnlearnMethod:
    return eval(name)