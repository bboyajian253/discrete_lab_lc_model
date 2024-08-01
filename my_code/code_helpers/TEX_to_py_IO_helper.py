import numpy as np
import sympy as sp
from latex2sympy2 import latex2sympy
import re

#a function to replace greek letters and other symbols from latex into things python likes
def replace_sym( myStr) :
    #the list of symbols and their replacements
    #use r"str_to_replace" here so py and re know to treat special characters as literals
    rep = {r"\alpha" : "a", r"\beta" : "B", r"\sigma" : "s", r"\phi" : "phi",
           r"_{t}" : "", r"_{t+1}" : "_prime", r"_{n}" : "_n"
           }
    #This line creates a new dictionary rep where each key is the escaped version of the original key k
    rep = dict((re.escape(k), v) for k, v in rep.items())
    #Here, re.compile() is used to compile a regular expression pattern of all the keys from the dictionary joined together 
    pattern = re.compile("|".join(rep.keys()))
    #uses the compiled regular expression pattern pattern to substitute matches in the myStr string with 
    #their corresponding values from the rep dictionary.
    #lambda m: rep[re.escape(m.group(0))] is a function that takes a match object m and returns the 
    #replacement string from the rep dictionary for the matched string m.group(0).
    newStr = pattern.sub(lambda m: rep[re.escape(m.group(0))], myStr)
    return newStr

def print_test_IO( myStr) :
    myF = myStr
    print("Printing the OG expression")
    print(myF)


    myF = replace_sym(myF)
    print("Printing after replacing symbols")
    print(myF)

    mySym = latex2sympy(myF)
    #print it in SymPy format (confirm reading of Tex)
    print("Printing as SymPy")
    print(mySym)

    print("Printing as re-converted TEX")
    print(sp.latex(mySym))


#Raw LaTeX
util_c_str = r"\alpha*c_{t}^{\alpha-1}l_{t}^{1-\alpha}\left(c_{t}^{\alpha}l_{t}^{1-\alpha}\right)^{-\sigma}"
util_l_str = r"(1-\alpha)*c_{t}^{\alpha}l_{t}^{-\alpha}\left(c_{t}^{\alpha}l_{t}^{1-\alpha}\right)^{-\sigma}"
c_star_str = r"\alpha\left[\frac{z_{t}^{H}}{\phi_{n}}\left(1-\phi_{H_{t}}\right)+\left(1+r\right)a_{t}-a_{t+1}\right]"
u_c_inv_str = r"\left(\frac{x}{a*l^{(1-a)*(1-s)}}\right)^{\frac{1}{a-1-s*a}}"
u_l_inv_str = r"\left(\frac{x*c^{-\alpha*(1-\sigma)}}{1-\alpha}\right)^{\frac{1}{\alpha*\sigma-\alpha-\sigma}}"
leis_giv_c_str = r"\frac{p}{z}*\frac{\left(1-\alpha\right)}{\alpha}*c"
util_c_giv_leis_str = r"\alpha c^{\alpha-1}l^{1-\alpha}\left(c^{\alpha}l^{1-\alpha}\right)^{-\sigma}"

print_test_IO(util_c_giv_leis_str)
