## Dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spStat
from pyColors import bcolors as bc
from typing import List, Dict, Union

gen_types:Dict[str,np.random.Generator] = {
    'default':np.random.MT19937,
    'PCG64':np.random.PCG64,
    'DXSM':np.random.PCG64DXSM
}

def initialize_generator(gen_type:str)->None:
    return np.random.Generator(gen_types[gen_type].__call__(seed = None))

def seed_generator(gen_type:str, seed:int)->None:
    return np.random.Generator(gen_types[gen_type].__call__(seed = seed))

def spawn_generator(generator:np.random.Generator)->List[np.random.Generator]:
    return generator.spawn(1)[0]

#%% Random Action
def draw_uniform(max:int, generator:np.random.Generator)->int:
    return generator.integers(low = 1,
                              high = max,
                              endpoint = True,
                              dtype = np.uint8)
    
# recursve exploder
def explode_uniform(max:int, generator:np.random.Generator, current_status:int = 0)->int:
    current_status += draw_uniform(max = max,
                                      generator = generator)
    if current_status % max == 0:
        return explode_uniform(max = max,
                               generator = generator,
                               current_status = current_status)
    return current_status

def draw_multiple_uniform(size:int, **kwargs):
    return np.array([
        draw_uniform(**kwargs)
        for _ in range(size)
    ])
    
def explode_multiple_uniform(size:int, **kwargs):
    return np.array([
        explode_uniform(**kwargs)
        for _ in range(size)
    ])
    
def draw_advantage_uniform(**kwargs):
    tmp = draw_multiple_uniform(size = 2, 
                                **kwargs)
    high, low = max(tmp), min(tmp)
    return high, low

def explode_advantage_uniform(**kwargs):
    tmp = explode_multiple_uniform(size = 2, 
                                   **kwargs)
    high, low = max(tmp), min(tmp)
    return high, low

#%% Formatter
def sum_formatter(values:Union[List, np.ndarray], modifier:int = 0)->str:
    out:str = f"{values[0]}"
    for value in values[1:]:
        out += f" + {value}"
    
    if modifier > 0:
        out += f" + {bc.BOLD}{bc.OKGREEN}{modifier}{bc.ENDC}"
        values[0] += modifier
    elif modifier < 0:
        out += f" - {bc.BOLD}{bc.FAIL}{modifier}{bc.ENDC}"
        values[0] += modifier
    
    out += f" = {sum(values)}"
    return out

def advantage_formatter(high, low, modifier:int = 0)->str:
    out = f"{bc.FAIL}{low}{bc.ENDC} | {bc.OKGREEN}{high}{bc.ENDC}"
    
    if modifier > 0:
        out += f" + {bc.BOLD}{bc.OKGREEN}{modifier}{bc.ENDC}"
        high += modifier
    elif modifier < 0:
        out += f" - {bc.BOLD}{bc.FAIL}{modifier}{bc.ENDC}"
        high += modifier
    
    out += f" = {high}"
    return out
        
def disadvantage_formatter(high, low, modifier:int = 0)->str:
    out = f"{bc.FAIL}{high}{bc.ENDC} | {bc.OKGREEN}{low}{bc.ENDC}"
    
    if modifier > 0:
        out += f" + {bc.BOLD}{bc.OKGREEN}{modifier}{bc.ENDC}"
        low += modifier
    elif modifier < 0:
        out += f" - {bc.BOLD}{bc.FAIL}{modifier}{bc.ENDC}"
        low += modifier
        
    out += f" = {low}"
    return out

#%% Helpers
def check_input(inp:Union[float, int])->bool:
    if int(inp) >= 2:
        return True
    return False

def format_print(**kwargs)->None:
    return print(sum_formatter(**kwargs))

def get_expectation(values:Union[List, np.ndarray])->float:
    return spStat.expectile(values)

def plot_distribution(type:str, sample:int = 10000, **kwargs):
    draw_fnc = {'uniform':draw_multiple_uniform,
                'exploding':explode_multiple_uniform}[type]
    
    sample = draw_fnc(size=sample,
                      **kwargs)
    expectile = get_expectation(sample)
    
    fig, ax = plt.subplots(1,1, figsize = (8,4), dpi = 100)
    ax.hist(np.array(sample),
            bins = max(sample))
    ax.axvline(expectile,
               label = f"ExP: {expectile:.2f}",
               color = 'k',
               ls = '--')
    
    # layout
    ax.set_title(f"{type} distribution")
    ax.set_xlabel('Draw Result [a.u.]')
    ax.set_ylabel('Counts [a.u.]')
    ax.legend(loc = 'upper right',
              fancybox = True,
              shadow = True)
    return fig, ax
