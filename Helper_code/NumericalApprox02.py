# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sympy as sp
from latex2sympy2 import latex2sympy

sp.init_printing()
y = sp.Symbol('y')

string1="\left(\frac{100}{y_{1}+y_{2}(y_{1})}\right)y_{1}-2y_{1}^{2}"
string2= "+2"

bigString = string1 + string2
#Raw LaTeX
#f = r"\frac{x}{1+x}"
#myF = r"\left(\frac{100}{y_{1}+y_{2}(y_{1})}\right)y_{1}-2y_{1}^{2}"
s1= r"0 = -\frac{100\left(\frac{1}{3}\left(-\frac{\sqrt[3]{2}\left(6y^{2}+\frac{15\sqrt{3}\left(16y^{3}+1350y\right)}{2\sqrt{4y^{4}+675y^{2}}}+675\right)y^{2}}{3\left(2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y\right)^{4/3}}+\frac{2\sqrt[3]{2}y}{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}+\frac{6y^{2}+\frac{15\sqrt{3}\left(16y^{3}+1350y\right)}{2\sqrt{4y^{4}+675y^{2}}}+675}{3\sqrt[3]{2}\left(2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y\right)^{2/3}}-2\right)+1\right)y}{\left(\frac{1}{3}\left(\frac{\sqrt[3]{2}y^{2}}{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}+\frac{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}{\sqrt[3]{2}}-2y\right)+y\right)^{2}}+\frac{100}{\frac{1}{3}\left(\frac{\sqrt[3]{2}y^{2}}{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}+\frac{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}{\sqrt[3]{2}}-2y\right)+y}-4y"
s2=r""
bigF = s1 

#Latex to SymPy
#sym = latex2sympy(f)
#mySym = latex2sympy(myF)
bigSym = latex2sympy(bigF)

#print it in SymPy format (confirm reading of Tex)
#print(sym)
#print(mySym)
#print(bigSym)
print("\n \n \n")
#print(sp.latex(bigSym))

#sp.solve(bigSym, y) #No algorithms are implemented to solve equation

f = sp.lambdify(y, bigSym)

guess = 2.0 #Set to 2 for best result
desiredRes = 0
epsilon = .0000001
inc =.0001
res = f(guess)
print(res)

print("entering loop")
while abs(res - desiredRes) >= epsilon:
    guess += inc
    print("\n")
    print("result is: " , res)
    print("guess is: " , guess)
    res = f(guess)
    
print("leaving loop")