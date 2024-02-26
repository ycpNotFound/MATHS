import sympy
import re
import math
import torch
from sympy.parsing.latex import parse_latex
from sympy import solve, parse_expr, Eq, symbols
from transformers import (
    AutoTokenizer
)

import functools
import signal


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator
            
@timeout(5)
def solve_equation_ns(expr_str):
    if 'x' not in expr_str:
        return None
    x = symbols('x') 
    equation = Eq(eval(expr_str.replace("=", "-")), 0)  
    result = solve(equation, x)[0]
    result = int(result)
    if result < 0:
        return None
    result_str = f'answer x = {result}'

    return result_str

class PluginProcessor():
    def __init__(self, tokenizer_path, num_beams, batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.plugin_bid = self.tokenizer.encode('<plugin>')[1]
        self.plugin_eid = self.tokenizer.encode('</plugin>')[1]
        self.num_beams = num_beams
        self.ids_to_add = [[] for i in range(num_beams*batch_size)]

    def calculate(self, expr_ids):
        # input expr_ids, output answer_ids
        expr_str = self.tokenizer.decode(expr_ids)
        # expr_str = expr_str.replace('>>', '')
        # if expr_str.split('=')[-1].strip() == '0':
        if '=' in expr_str and expr_str.split('=')[-1] != '':
            try:    
                result = solve_equation_ns(expr_str)
            except:
                result = None
        else:
            expr_str = expr_str.replace(' ', '')
            if 'V' in expr_str.split('=')[0]:
                expr_str = expr_str.split('=')[-1]
            try:
                expr_str = re.sub(r'\\sqrt', 'math.sqrt', expr_str)
                expr_str = expr_str.replace('^', '**')
                
                # expr = parse_expr(expr_str)
                result = eval(expr_str)
                result = str(result)[:5]
                if '.0' in result:
                    result = result.split('.')[0]
                if '.' in result:
                    result = None
            except:
                result = None
            # result = f'={result}'
        if result == None:
            return []
        res_ids = self.tokenizer.encode(result)[1:-1]
        return res_ids
    

    def __call__(self, input_ids, next_token_scores):
        # set logits at next token idx (to 0)x to the same, other idx to -100
        for i, ids_to_add_beam_i in enumerate(self.ids_to_add):
            if len(ids_to_add_beam_i) == 0: 
                continue
            next_idx = ids_to_add_beam_i.pop(0)
            # next_token_scores[i, next_idx] = 0
            next_token_scores[i, next_idx] = next_token_scores[i].max() + 1
            next_token_scores[i, :next_idx] = -10
            next_token_scores[i, next_idx+1:] = -10

        # find </plugin> and add ids
        for i, inputs_beam_i in enumerate(input_ids):
            if inputs_beam_i[-1] != self.plugin_eid:
                continue
            last_begin_index = torch.where(inputs_beam_i == self.plugin_bid)[0]
            if len(last_begin_index) == 0:
                continue
            last_begin_index = last_begin_index[-1]
            end_indexs = torch.where(inputs_beam_i == self.plugin_eid)[0]
            # check the situation: <p>...</p>...</p>
            if len(end_indexs) >= 2:
                last_end_index = end_indexs[-2]
                if last_end_index > last_begin_index:
                    continue
            expr_ids = inputs_beam_i[last_begin_index+1: -1]
            try:
                ans_ids = self.calculate(expr_ids)
                if ans_ids:
                    self.ids_to_add[i] = self.ids_to_add[i] + ans_ids
            except:
                # print(f'parse latex error: {self.tokenizer.decode(expr_ids)}')
                pass

        return next_token_scores

    def update_ids_to_add(self, next_indices):
        ids_to_add_new = []
        for index in next_indices:
            ids_to_add_new.append(self.ids_to_add[index])
        self.ids_to_add = ids_to_add_new
        

def test_for_calculate():
    expr_str = ' 24  /  x - 10 = 2'

    result = solve_equation_ns(expr_str)
    print(result)



if __name__ == "__main__":
    test_for_calculate()