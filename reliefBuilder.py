# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
from numpy import ones
from mpl_toolkits.mplot3d import Axes3D

a, b = 3e-6, 6e-6 # ширина і висота перерізу транзистора
Nd = 2.1e18**(2/3) # концентрація донорів у перерізі
Np = 200 # кількість квазічастинок
M = Nd/Np # коефіцієнт помноження

h = 1e-7 #просторовий крок
Nx = int(a/h) # кількість клітинок по висоті
Ny = int(b/h) # кількість клітинок по ширині

Nt = 100 # кількість часових кроків
time = 1 # час моделювання
dt = time/Nt #часовий крок

v = 2e5 # середня швидкість носіїв за вказаних потенціалів

#напруги:
Udrain  = 1e-10
Usourse = 5
Ugate = -2

#часові матриці координат
import numpy as np
x = np.int_(np.zeros((Np, Nt)))
y = np.int_(np.zeros((Np, Nt)))
#на першому часовому кроці задамо випадковими
x[:,0] =  np.int_(np.random.randint(0, Nx-1, size=(Np)))
y[:,0] =  np.int_(np.random.randint(0, Ny-1, size=(Np)))

# вектор значень ефективних мас (в Г - долині)
m = np.ones((Np))*9.1e-31

#часові матриці швидкостей
vx = np.int_(np.zeros((Np, Nt)))
vy = np.int_(np.zeros((Np, Nt)))
#на першому часовому кроці задамо випадковими
vx[:,0] =  np.int_(np.random.randint(-2, 2, size=(Np)))*0.1*v + v
vy[:,0] =  np.int_(np.random.randint(-2, 2, size=(Np)))*0.1*v + v

#в часову матрицю нижче будемо записувати факт перетину i-м електроном області витоку
particlecounter = np.zeros((Nt))

def potential_finder(Nx,Ny,Uz,Uv,Uc,eps):
    '''функція для знаходження виду потенціального рель'єфа структури №1'''
    
    V  = ones((Nx,Ny))*1e-3    # сітка потенціалів
    V_old = ones((Nx,Ny))*1e-3 # сітка для збереження потенціалів
    width = round(Ny/6)     #довжина l1 = l2 = l3/2 = l4 = l5 у відліках
    accuracy = list([100])  # точнысть отриманого результата
    
    #т.к. "первый" отсчет в человеческом понимании - это "нулевой" отсчет
    NX = Nx - 1
    NY = Ny - 1 
    
    counter = 0
    while(True):
        for i in range (0,Nx):
            for j in range (0,Ny):
                
                #електроди
                if i == 0 and 0 <= j <= width:
                    V[i,j] = Uc
                elif i == 0 and 2*width <= j <= 4*width:
                    V[i,j] = Uz
                elif i == 0 and j>=5*width:
                    V[i,j] = Uv
                
                # міжелектродний простір
                if i == 0 and width < j < 2*width:
                    V[i,j] = (V[i,j-1] + V[i,j+1] + 2*V[i+1,j])/4
                if i == 0 and 4*width < j < 5*width:
                    V[i,j] = (V[i,j-1] + V[i, j+1] + V[i+1,j])/4
                
                #ліва стінка
                if 0 < i < NX and j == 0:
                    V[i,j] = (V[i-1,j] + V[i+1,j] + 2*V[i,j+1])/4
                
                #права стінка
                if 0 < i < NX and j == NY:
                    V[i,j] = (V[i-1,j] + V[i+1,j] + 2*V[i,j-1])/4
                
                #дно лівий кут
                if i == NX and j == 0:
                    V[i,j] = (2*V[i-1,j] + 2*V[i,j+1])/4
                
                #дно правий кут
                if i == NX and j == NY:
                    V[i,j] = (2*V[i-1,j] + 2*V[i,j-1])/4
                
                #дно
                if i == NX and 0 < j < NY:
                    V[i,j] = (V[i,j-1] + V[i,j+1] + 2*V[i-1,j])/4
                
                #середина
                if 0 < i < NX and 0 < j < NY:
                    V[i,j] = (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])/4
        
        # при першому прохожденні сенсу оцінювати похибку немає, адже вибрані 
        #випадкові ПУ
        if counter != 0:
            #умова переривання розрахунку
            accuracy.append(float((abs(np.divide(np.subtract(V,V_old),V)) *100).max()))
            if  accuracy[counter] < eps:
                break
            
        counter = counter + 1
        V_old = np.matrix(V)
    return V

# розв`яжемо рівняння Лапласа
def voltage(Nx,Ny,Udrain,Usourse,Ugate):
    '''функція для розв'язання рівняння Лапласа
    (знаходження потенціального рельєфу)
    {використано метод послідовних наближень}'''
    
    import numpy as np
    #початкові наближення:
    V, V_old,= np.ones((Nx,Ny))*0.1, np.ones((Nx,Ny))*0.1
    
    width                   = int((Ny-1)/6) # l1 = l2...
    V[0,0:width+1]          = Usourse       # стік
    V[0,2*width:4*width+1]  = Ugate         # затвор
    V[0,5*width:]           = Udrain        # витік
    
    #допоки похибка не буде = 0.01%, продовжувати обчислення:
    while(np.max(np.abs(np.divide(np.multiply(np.subtract(V,V_old),100),V_old))) > 1):
        
        #збережемо обраховану матрицю для порівняння з наступною:
        V_old = np.matrix(V)
        # права стінка:
        V[1:Nx-1,Ny - 1] = (V[0:Nx-2,Ny - 1] + V[2:Nx,Ny - 1] + V[1:Nx-1,Ny - 2]*2)/4
        # дно:
        V[Nx-1,1:Ny-1] = (V[Nx-1,0:Ny-2] + V[Nx-1,2:Ny] + V[Nx-2,1:Ny-1]*2)/4
        #область між стоком і затвором:
        V[0,width+1:2*width] = (V[0,width:2*width-1] + V[0,width+2:2*width+1] + V[1,width+1:2*width]*2)/4
        # область між затвором і витоком:
        V[0,4*width+1:5*width] = (V[0,4*width:5*width-1] + V[0,4*width+2:5* width+1] + V[1,4*width+1:5*width]*2)/4
        #ліва стінка:
        V[1:Nx-1,0] = (V[0:Nx-2,0] + V[2:Nx,0] + V[1:Nx-1,1]*2)/4
        #лівий нижній кут:
        V[Nx-1,0] = (V[Nx-2,0]*2 + V[Nx-1,1]*2)/4
        #правий нижній кут:
        V[Nx-1,Ny-1] = (V[Nx-1,Ny-2]*2 + V[Nx-2,Ny-1]*2)/4
        #середина:
        V[1:Nx-1,1:Ny-1] = (V[0:Nx-2, 1:Ny-1] + V[2:Nx, 1:Ny-1] + V[1:Nx-1,0:Ny-2] + V[1:Nx-1,2:Ny])/4
    
    return V

def electricfield(V, Nx, Ny, h):
    ''' функція для знаходження напруженості електричного поля'''
    import numpy as np
    Ex = np.zeros((Nx,Ny))
    Ey = np.zeros((Nx,Ny))
    Ex[0:Nx-1,0:Ny-1] = (V[0:Nx-1,0:Ny-1] - V[1:Nx,0:Ny-1] + V[0:Nx-1,1:Ny] - V[1:Nx,1:Ny])/(2*h)
    Ey[0:Nx-1,0:Ny-1] = (V[0:Nx-1,0:Ny-1] - V[0:Nx-1,1:Ny] + V[1:Nx,0:Ny-1] - V[1:Nx,1:Ny])/(2*h)
    
    return Ex,Ey

def relief_build (V,Nx,Ny):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(60, 35) #3d picture rotation
    xsample = np.linspace(0, Nx-1, Nx)
    ysample = np.linspace(0, Ny-1, Ny)
    Xsample, Ysample = np.meshgrid(ysample, xsample)
    ax.plot_wireframe(Ysample,Xsample,V)
    plt.show()
    
def vector_field_buid(Ex,Ey):
    plt.figure()
    plt.quiver(Ex,Ey)
    plt.show()

V = potential_finder(Nx,Ny,Ugate,Udrain,Usourse,0.01)
#V = voltage(Nx,Ny,Udrain,Usourse,Ugate)
Ex,Ey = electricfield(V, Nx, Ny, h)
relief_build (V,Nx,Ny)
vector_field_buid(Ex,Ey)
relief_build (Ex,Nx,Ny)
relief_build (Ey,Nx,Ny)
