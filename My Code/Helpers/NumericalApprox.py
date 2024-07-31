# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sympy as sp
from latex2sympy2 import latex2sympy

sp.init_printing()
y = sp.Symbol('y')

def myTexToSymPy(tex):
    sPy = latex2sympy(tex)
    print("Tex to Sympy gives: ", sPy)
    return sPy

def numApprox(guess, desRes, eps, incr, func):
   res = func(guess)
   print("entering loop")
   
   while abs(res - desRes) >= eps:
       guess -= incr #can be changed to -=guess to check vals < guess instead of vals > guess
       print("\n")
       print("result is: " , res)
       print("guess is: " , guess)
       res = func(guess)
 
   print("leaving loop")

def testRun(bF):
    print("First explicitly \n")

    #Latex to SymPy
    bigSym = latex2sympy(bF)

    #print it in SymPy format (confirm reading of Tex)
    print(bigSym)
    #print("\n \n \n")
    #print(sp.latex(bigSym))

    f = sp.lambdify(y, bigSym)

    guess = 2.0 #Set to 2 for best result
    desiredRes = 0
    epsilon = .0000001
    inc =.0001
    res = f(guess)
    print(res)

    print("entering loop")
    while abs(res - desiredRes) >= epsilon:
        guess -= inc
        print("\n")
        print("result is: " , res)
        print("guess is: " , guess)
        res = f(guess)
    
    print("leaving loop \n")

#******Main*****
#Raw Latex
bigF = r"0 = -\frac{100\left(\frac{1}{3}\left(-\frac{\sqrt[3]{2}\left(6y^{2}+\frac{15\sqrt{3}\left(16y^{3}+1350y\right)}{2\sqrt{4y^{4}+675y^{2}}}+675\right)y^{2}}{3\left(2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y\right)^{4/3}}+\frac{2\sqrt[3]{2}y}{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}+\frac{6y^{2}+\frac{15\sqrt{3}\left(16y^{3}+1350y\right)}{2\sqrt{4y^{4}+675y^{2}}}+675}{3\sqrt[3]{2}\left(2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y\right)^{2/3}}-2\right)+1\right)y}{\left(\frac{1}{3}\left(\frac{\sqrt[3]{2}y^{2}}{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}+\frac{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}{\sqrt[3]{2}}-2y\right)+y\right)^{2}}+\frac{100}{\frac{1}{3}\left(\frac{\sqrt[3]{2}y^{2}}{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}+\frac{\sqrt[3]{2y^{3}+15\sqrt{3}\sqrt{4y^{4}+675y^{2}}+675y}}{\sqrt[3]{2}}-2y\right)+y}-4y"
testF =r"y^{2}-9" #test

#Raw Latex to SymPy
#mySym = myTexToSymPy(testF)#test
mySym = myTexToSymPy(bigF)

#Create sympy function of y
f = sp.lambdify(y, mySym)

#Approximate solution
#testRun(bigF)
#print("Now with functions \n")
numApprox(3.0, 0.0, .0001, .0001, f)

 
    
    
    