from typing import Any, Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor


from functools import reduce, wraps
from time import perf_counter
from typing import Callable



def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        return results, run_time
    return wrapper

@timer
def p_map(items: List[Any], func: Callable, options: Optional[dict[str, any]] = None) -> Any:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, items))
    return results

# def compose(f, g):
#     return lambda *args, **kwargs: f(g(*args, **kwargs))

def composite_function(*func):
    def compose(f, g):
        return lambda x : f(g(x))              
    return reduce(compose, func, lambda x : x)

def compose(*functions):
    """
    Composes multiple functions into a single function from right to left.
    """
    return reduce(lambda f, g: lambda x: f(g(x)), reversed(functions))

def pipe(*functions):
    """
    Composes multiple functions into a single function from left to right.
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


