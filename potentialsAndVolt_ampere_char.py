#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:49:21 2019

@author: serhiibahdasaryants
"""
a, b = 3e-6, 6e-6 # ширина і висота перерізу транзистора
Nd = 2.1e18**(2/3) # концентрація донорів у перерізі
Np = 200 # кількість квазічастинок
M = Nd/Np # коефіцієнт помноження

h = 1e-8/2 #просторовий крок
Nx = int(a/h) # кількість клітинок по висоті
Ny = int(b/h) # кількість клітинок по ширині

Nt = 10000 # кількість часових кроків
time = 1 # час моделювання
dt = time/Nt #часовий крок

v = 2e5 # середня швидкість носіїв за вказаних потенціалів

#напруги:
Udrain  = 1e-10
Usourse = 5

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


# розв`яжемо рівняння Лапласа
def voltage(Nx,Ny,Udrain,Usourse,Ugate):
    '''функція для розв'язання рівняння Лапласа
    (знаходження потенціального рельєфу)
    {використано метод послідовних наближень}'''
    
    import numpy as np
    #початкові наближення:
    V, V_old,= np.ones((Nx,Ny))*0.1, np.ones((Nx,Ny))*0.1e-10
    
    width                   = int((Ny-1)/6) # l1 = l2...
    V[0,0:width+1]          = Usourse       # стік
    V[0,2*width:4*width+1]  = Ugate         # затвор
    V[0,5*width:]           = Udrain        # витік
    
    #допоки похибка не буде = 0.01%, продовжувати обчислення:
    while(np.max(np.abs(np.divide(np.multiply(np.subtract(V,V_old),100),V_old))) > 0.1):
        
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
        print(np.max(np.abs(np.divide(np.multiply(np.subtract(V,V_old),100),V_old))))
    return V

def electricfield(V, Nx, Ny, h):
    ''' функція для знаходження напруженості електричного поля'''
    import numpy as np
    Ex = np.zeros((Nx,Ny))
    Ey = np.zeros((Nx,Ny))
    Ex[0:Nx-1,0:Ny-1] = (V[0:Nx-1,0:Ny-1] - V[1:Nx,0:Ny-1] + V[0:Nx-1,1:Ny] - V[1:Nx,1:Ny])/(2*h)
    Ey[0:Nx-1,0:Ny-1] = (V[0:Nx-1,0:Ny-1] - V[0:Nx-1,1:Ny] + V[1:Nx,0:Ny-1] - V[1:Nx,1:Ny])/(2*h)
    
    return Ex,Ey

def electricfield1(V):
    '''функція для знахождення напруженості електричного поля'''
    import numpy as np
    E = np.multiply(np.gradient(V), -1/h)
    Ex, Ey = E[0], E[1]
    return Ex,Ey

def relief_build (V,Nx,Ny):
    '''функція для побудови 3d поверхні'''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(60, 35) #3d picture rotation
    xsample = np.linspace(0, Nx-1, Nx)
    ysample = np.linspace(0, Ny-1, Ny)
    Xsample, Ysample = np.meshgrid(ysample, xsample)
    ax.plot_wireframe(Ysample,Xsample,V)
    plt.show()
    
def vector_field_buid(Ex,Ey):
    '''функція для побудови векторного поля'''
    import matplotlib.pyplot as plt
    plt.figure()
    plt.quiver(Ex,Ey)
    plt.show()
    
    
def speedcoordinate(x,y,vx,vy,k,dt,m,Ex,Ey):
    '''функція для знаходження координат і швидкостей частинки на k-му часовому кроці'''
    q = 1.6e-19
    #координати частинки:
    x[:,k] = x[:,k-1] + vx[:,k-1]*dt - np.multiply(q*dt*dt*h/(2*m[:]), Ex[x[:,k-1], y[:,k-1]])
    y[:,k] = y[:,k-1] + vy[:,k-1]*dt - np.multiply(q*dt*dt*h/(2*m[:]), Ey[x[:,k-1], y[:,k-1]])
    #швидкості частинки:
    vx[:,k] = vx[:,k-1] - np.multiply(q*dt*h/m[:], Ex[x[:,k-1], y[:,k-1]])
    vy[:,k] = vy[:,k-1] - np.multiply(q*dt*h/m[:], Ey[x[:,k-1], y[:,k-1]])
    
    return np.int_(x),np.int_(y),vx,vy


def intersection(pt1, pt2, ptA, ptB, Nx, Ny):
    '''функція пошуку перетину двох відрізків, які задані двома точками кожний'''
    from scipy.optimize import newton
    x1, y1, x2, y2 = float(pt1[0]), float(pt1[1]), float(pt2[0]), float(pt2[1]) #координати першого відрізку
    x3, y3, x4, y4 = float(ptA[0]), float(ptA[1]), float(ptB[0]), float(ptB[1]) #координати другого відрізку
        
    fx = lambda x: (x-x1)*(y2-y1)*(x4-x3) + (x2-x1)*(x4-x3)*(y1-y3) - (x-x3)*(y4-y3)*(x2-x1) # прирівнюємо ординати двох відрізків
    fy = lambda y: (y-y1)*(x2-x1)*(y4-y3) + (y2-y1)*(y4-y3)*(x1-x3) - (y-y3)*(x4-x3)*(y2-y1) # прирівнюємо абсциси двох відрізків
    

    x_ = int(newton(fx,x1)) # знаходимо абсцису перетину
    y_ = int(newton(fy,y1)) # знаходимо ординату перетину
    
    # накладаємо умови, що точка перетину повинна лежати на обох відрізках
    if (min(x1,x2) <= x_ <= max(x1,x2) and min(y1,y2) <= y_ <= max(y1,y2)) and (min(x3,x4) <= x_ <= max(x3,x4) and min(y3,y4) <= y_ <= max(y3,y4)):
        return np.array([x_,y_,1])
    else:
        # якщо точка не належить жодному з відрізків
        return np.array([1e9,1e9,0])

def whereyougo(pt1, pt2, Nx, Ny):
    '''функія для визначення області яку перетнула частинка (або знахождення 
    частинки в середині транзистора) '''
    width = round((Ny-1)/6)     #довжина l1 = l2 = l3/2 = l4 = l5 у відліках
    #т.к. "первый" отсчет в человеческом понимании - это "нулевой" отсчет:
    NX = Nx - 1
    NY = Ny - 1
    
    pleft1, pleft2      = (0,0), (NX,0)             # ліва границя
    pright1, pright2    = (0,NY), (NX,NY)           # права границя
    pbottom1, pbottom2  = (NX,0), (NX,NY)           # дно
    pstik1,pstik2       = (0,0), (0,width)          # область стоку
    pzatvor1, pzatvor2  = (0,2*width), (0,4*width)  # область затвора
    pvytik1,pvytik2     = (0,5*width), (0,NY)       # область витоку
    pobl11,pobl12       = (0,width), (0,2*width)    # область між стоком і затвором
    pobl21,pobl22       = (0,4*width), (0,5*width)  # область між затвором і витоком
    
    #знайдемо з якою з областей відбувся перетин
    left    = intersection(pt1, pt2, pleft1,   pleft2,   Nx,Ny)
    right   = intersection(pt1, pt2, pright1,  pright2,  Nx,Ny)
    bottom  = intersection(pt1, pt2, pbottom1, pbottom2, Nx,Ny)
    stik    = intersection(pt1, pt2, pstik1,   pstik2,   Nx,Ny)
    zatvor  = intersection(pt1, pt2, pzatvor1, pzatvor2, Nx,Ny)
    vytik   = intersection(pt1, pt2, pvytik1,  pvytik2,  Nx,Ny)
    obl1    = intersection(pt1, pt2, pobl11,   pobl12,   Nx,Ny)
    obl2    = intersection(pt1, pt2, pobl21,   pobl22,   Nx,Ny)
    
    if left[-1]     == 1:
        igothere    =  1#left
    elif right[-1]  == 1:
        igothere    =  2#right
    elif bottom[-1] == 1:
        igothere    =  3#bottom
    elif stik[-1]   == 1:
        igothere    =  4#stik
    elif zatvor[-1] == 1:
        igothere    =  5#zatvor
    elif vytik[-1]  == 1:
        igothere    =  6#vytik
    elif obl1[-1]   == 1:
        igothere    =  7#obl1
    elif obl2[-1]   == 1:
        igothere    =  8#obl2
    else:
        igothere    =  9#middle
    
    return igothere

def speedcoordinatecontrol(x,y,vx,vy,k,n,Nx,Ny):
    '''функція для визначення того, куди пересунулась кожна з частинок'''
    import numpy as np
    #вектор для запису положення кожної частнки в просторі
    govector = np.zeros((Np))
    
    for i in range(len(x)):
        
        if n == 0: #якщо тригер виключений, то минулими координатами вважаємо x[i,k-1], y[i,k-1]
            x_old, y_old = x[i,k-1], y[i,k-1]
        else: # якщо тригер увімкнуто, тобто n != 0, то минулі координати задаються в центрі транзистора
            x_old, y_old = (Nx-1)/2, (Ny-1)/2
        
        govector[i]= whereyougo((x_old, y_old), (x[i,k], y[i,k]), Nx, Ny)
    return govector

def wallbounce(govector,x,y,vx,vy,k,Ny,Nx,Np,Nt,v):
    '''функція реалізації пружного відскоку частинки від стінки'''
    import numpy as np
    width = (Ny-1)/6
    
    for i in range(len(govector)):
        if govector[i] == 1:
            y[i,k]  = abs(y[i,k])+1
            vy[i,k] = - vy[i,k]
            x[i,k]  = abs(x[i,k])
            
        elif govector[i] == 2:
            y[i,k]  = Ny-1 - abs(y[i,k] - Ny-1)
            vy[i,k] = - vy[i,k]
            x[i,k]  = abs(x[i,k])
            
        elif govector[i] == 3:
            x[i,k]  = Nx-1 - abs(x[i,k] - Nx-1)
            vx[i,k] = -vx[i,k]
            y[i,k]  = abs(y[i,k])
            
        #якщо електрон попадає на стік або затвор, або витік то нехай він з'являється поблизу області стоку:
        elif govector[i] == 4 or govector[i] == 5 or govector[i] == 6:
            # запишимо факт того, що частинка перетниула витік:
            if govector[i] == 6:
                particlecounter[k] = particlecounter[k] + 1
                
            x[i,k]  = np.random.randint(1, width)
            y[i,k]  = np.random.randint(1, width)
            vx[i,k] = np.random.randint(-2, 2)*0.1*v + v
            vy[i,k] = np.random.randint(-2, 2)*0.1*v + v
            
        elif govector[i] == 7 or govector[i] == 8:
            x[i,k]  = abs(x[i,k])+1
            vx[i,k] = -vx[i,k]
            y[i,k] = abs(y[i,k])
    return x,y,vx,vy
            

def whichdispersal(vx,vy,Np,Nt,m,k):
    '''функція для визначення типу розсіяння або його відсутності'''
    import numpy as np
    m0 = 9.1e-31 # маса електрона
    v_avarage = np.zeros((Np))
    v_avarage = np.sqrt(np.multiply(vx[:,k],vx[:,k]) + np.multiply(vy[:,k],vy[:,k])) # вектор усереднених швидкостей
    energy    = np.multiply(np.multiply(v_avarage,v_avarage),m.reshape(Np))/2
    
    energy = energy*6.242*1e18 # переведемо в електронвольти
    dispersal = np.zeros((Np)) # вектор, що містить назву розсіювання для і-ї частинки

    for i in range(Np): # цикл по частинках
        
        which = float(np.random.rand(1)) # випадкове число від 0 до 1 (float)
        
        if m[i] <= m0:  # визначимо в якій долині знаходиться частинка
            valley = 'Г'
        else:
            valley = 'L'
            
        if valley == 'Г': # якщо електрон належить Г - долині
        
            if 0 <= energy[i] <= 0.2: #якщо його енергії лежать в даному діапазоні
                # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.02:
                    dispersal[i] = 1#'Ac'
                elif 0.02 < which <= 0.1:
                    dispersal[i] = 2#'POe'
                elif 0.1 < which <= 0.11:
                    dispersal[i] = 3#'POa'
                elif 0.11 < which <= 0.12:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 1.9*m0
                elif 0.12 < which <= 0.2:
                    dispersal[i] = 5#'Imp'
                elif which > 0.2:
                    dispersal[i] = 6#'No'
                
            elif 0.2 < energy[i] <= 0.4:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.02:
                    dispersal[i]  = 1#'Ac'
                elif 0.02 < which <= 0.1:
                    dispersal[i] = 2#'POe'
                elif 0.1 < which <= 0.15:
                    dispersal[i] = 3#'POa'
                elif 0.15 < which <= 0.21:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 1.9*m0
                elif 0.21 < which <= 0.235:
                    dispersal[i] = 5#'Imp'
                elif which > 0.235:
                    dispersal[i] = 6#'No'
        
            elif 0.4 < energy[i] <= 0.6:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.02:
                    dispersal[i] = 1#'Ac'#Ac
                elif 0.02 < which <= 0.1:
                    dispersal[i] = 2#'POe'#POe
                elif 0.1 < which <= 0.2:
                    dispersal[i] = 3#'POa'
                elif 0.2 < which <= 0.4:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 1.9*m0
                elif 0.4 < which <= 0.41:
                    dispersal[i] = 5#'Imp'
                elif which > 0.41:
                    dispersal[i] = 6#'No'
                
            elif 0.6 < energy[i] <= 0.8:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.02:
                    dispersal[i] = 1#'Ac'#Ac
                elif 0.02 < which <= 0.1:
                    dispersal[i] = 2#'POe'
                elif 0.1 < which <= 0.225:
                    dispersal[i] = 3#'POa'
                elif 0.225 < which <= 0.65:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 1.9*m0
                elif 0.65 < which <= 0.7:
                    dispersal[i]= 5#'Imp'
                elif which > 0.7:
                    dispersal[i] = 6#'No'
                
            elif 0.8 < energy[i] <= 1:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.02:
                    dispersal[i] = 1#'Ac'#Ac
                elif 0.02 < which <= 0.1:
                    dispersal[i] = 2#'POe'
                elif 0.1 < which <= 0.24:
                    dispersal[i] = 3#'POa'
                elif 0.24 < which <= 0.8:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 1.9*m0
                elif 0.8 < which <= 0.9:
                    dispersal[i] = 5#'Imp'
                elif which > 0.9:
                    dispersal[i] = 6#'No'
                    
        elif valley == 'L': # якщо електрон належить L - долині
            
            if 0 <= energy[i] <= 0.2: #якщо його енергії лежать в даному діапазоні
                # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.025:
                    dispersal[i] = 1#'Ac'#Ac
                elif 0.025 < which <= 0.035:
                    dispersal[i] = 2#'POe'
                elif 0.035 < which <= 0.1:
                    dispersal[i] = 3#'POa'
                elif 0.1 < which <= 0.275:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 0.067*m0
    
                elif which > 0.275:
                    dispersal[i] = 6#'No'
                
            elif 0.2 < energy[i] <= 0.4:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.03:
                    dispersal[i] = 1#'Ac'
                elif 0.03 < which <= 0.04:
                    dispersal[i] = 2#'POe'
                elif 0.04 < which <= 0.105:
                    dispersal[i] = 3#'POa'
                elif 0.105 < which <= 0.59:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 0.067*m0
    
                elif which > 0.59:
                    dispersal[i] = 6#'No'
        
            elif 0.4 < energy[i] <= 0.6:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.035:
                    dispersal[i] = 1#'Ac'
                elif 0.035 < which <= 0.045:
                    dispersal[i] = 2#'POe'
                elif 0.045 < which <= 0.11:
                    dispersal[i] = 3#'POa'
                elif 0.11 < which <= 0.75:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 0.067*m0
    
                elif which > 0.75:
                    dispersal[i] = 6#'No'
                
            elif 0.6 < energy[i]<= 0.8:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.04:
                    dispersal[i] = 1#'Ac'
                elif 0.04 < which <= 0.05:
                    dispersal[i] = 2#'POe'
                elif 0.05 < which <= 0.115:
                    dispersal[i] = 3#'POa'
                elif 0.115 < which <= 0.9:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 0.067*m0
    
                elif which > 0.9:
                    dispersal[i] = 6#'No'
                
            elif 0.8 < energy[i] <= 1:#якщо його енергії лежать в даному діапазоні
            # визначимо вид розсіювання імовірнісним експериментом
                if 0<= which <= 0.05:
                    dispersal[i] = 1#'Ac'
                elif 0.05 < which <= 0.06:
                    dispersal[i] = 2#'POe'
                elif 0.06 < which <= 0.12:
                    dispersal[i] = 3#'POa'
                elif 0.12 < which <= 1:
                    dispersal[i] = 4#'Г-L'
                    m[i] = 0.067*m0
    
    energy = energy/6.24*1e-18
    return dispersal


def diffusion_convert(dispersal, x, y, v, vx, vy, m, Np,k, Ny):
    'функція яка змінює швидкості в залежності від типу розсіяння'
    from numpy import sqrt
    from numpy.random import rand
    
    width = (Ny-1)/6
    for i in range(Np): #цикл по частинках
        
        #врахуэмо процеси генерації/рекомбінації носіїв
        genrec = np.random.rand(1)
        if genrec >= 0.99:
            x[i,k]  = np.random.randint(1, width)
            y[i,k]  = np.random.randint(1, width)
            vx[i,k] = np.random.randint(-2, 2)*0.1*v + v
            vy[i,k] = np.random.randint(-2, 2)*0.1*v + v
            continue
        else:
            pass
        
        xORy = rand(1) # для "розвертання" або x-ої, або у-ої складової швидкості
        
        if dispersal[i] == 1 or dispersal[5] == 1: # розсіювання Ac та Imp
            #пружний відскок частинки
            if xORy <= 0.5:
                vx[i,k] = -vx[i,k]
            else:
                vy[i,k] = - vy[i,k]
                
        elif dispersal[i] == 2 or dispersal[i] == 3:# розсіювання POe та POa
            
            #обрахуємо енергію частинки
            if xORy >= 0.5:
                energy = m[i]*vx[i,k]*vx[i,k]/2
            else:
                energy = m[i]*vy[i,k]*vy[i,k]/2
            
            Eplus_minus = rand(1) #забрати або додати енергію
            
            if Eplus_minus >=0.5:
                energy  = energy - 0.03*1.6e-19
            else:
                energy  = energy + 0.03*1.6e-19
            
            if xORy >= 0.5:
                if energy >= 0: #якщо значення енергії >= 0 
                    vx[i,k] = sqrt(2*energy/m[i])
                else:
                    vx[i,k] = - sqrt(abs(2*energy/m[i]))
            else:
                if energy >= 0: #якщо значення енергії >= 0 
                    vy[i,k] = sqrt(2*energy/m[i])
                else:
                    vy[i,k] = - sqrt(abs(2*energy/m[i]))
        
        elif dispersal[i] == 4: # розсіювання на переході Г-L
            xORy = rand(1)# для передачі енергії або x-ій, або у-ій складовій швидкості 
            
            if xORy <= 0.5:
                energy = m[i]*vx[i,k]*vx[i,k]/2
            else:
                energy = m[i]*vy[i,k]*vy[i,k]/2
                
            Eplus_minus = rand(1) #забрати або додати енергію
            
            if Eplus_minus >=0.5:
                energy  = energy - 0.33*1.6e-19
            else:
                energy  = energy + 0.33*1.6e-19
            
            if xORy <= 0.5:
                if energy < 0: #якщо значення енергії < 0 
                    vx[i,k] = - sqrt(abs(2*energy/m[i]))
                else:
                    vx[i,k] = sqrt(2*energy/m[i])
            else:
                if energy < 0: #якщо значення енергії < 0 
                    vy[i,k] = - sqrt(abs(2*energy/m[i]))
                else:
                    vy[i,k] = sqrt(2*energy/m[i])
                
        elif dispersal[i] == 6: #немає розсіювання
            pass # ніяк не змінювати
            
        # врахуємо кут розсіювання (нехай він буде не ідеальний)
        # втрат в енергії на даному етапі бути не повинно, т.я. ми просто змінюємо
        # напрям вектору, а не його величину
        
        velocity_trigger = rand(1)# вибір від якої швидкості беремо частину енергії 
        # і якій будемо передаввати
        if velocity_trigger >= 0.5:
            whose_part = vx[i,k]
        else:
            whose_part = vy[i,k]
        
        #пружне розсіювання:
        vxORvy = rand(1) # вибираємо до якої швидкості будемо додавати 
        
        if dispersal[i] == 1 or dispersal[i] == 5:

            if vxORvy >= 0.5:
                to_add = whose_part*0.01
            else:
                to_add = - whose_part*0.01
            
            vx[i,k] = vx[i,k] + to_add
            vy[i,k] = vy[i,k] - to_add
        
        #непружне розсіювання:
        elif dispersal[i] == 2 or dispersal[i] ==3 or dispersal[i] == 4:
            
            if vxORvy >= 0.5:
                to_add = whose_part*0.05
            else:
                to_add = - whose_part*0.05
            
            vx[i,k] = vx[i,k] + to_add
            vy[i,k] = vy[i,k] - to_add
        
        
    return x, y, vx, vy 


#задамо напруги на затворі
Voltages = np.array([2.0,1.0,0.0])
#у вектор нижче запишемо скільки частинок перетнуло область стоку при кожній з напруг
particles_on_each_itteration = np.zeros((len(Voltages)))
voltagecounter = 0

for Ugate in Voltages:
    print('Voltage = ' + str(Voltages[voltagecounter]))
    #знаходимо вигляд потенціального рел'єфу:
    V = voltage(Nx,Ny,Udrain,Usourse,Ugate)
    #знаходимо вигляд напруженості електричного поля:
    Ex,Ey = electricfield(V, Nx, Ny, h)
    
    #знайдемо швидкість і координати частинок на кожному кроці:
    for k in range(1,Nt/2): # in ideal there should be Nt instead of Nt/2
        print(k)
        #знайдемо коодинати і швидкості всіх частинок через певний часовий крок k:
        x,y,vx,vy = speedcoordinate(x,y,vx,vy,k,dt,m,Ex,Ey)
        #знайдемо куди перемістилась кожна з частинок:
        govector  = speedcoordinatecontrol(x,y,vx,vy,k,0,Nx,Ny)
        #усі частинки повинні знаходитись в області моделювання
        while (min(govector[:]) != 9):
            x,y,vx,vy = wallbounce(govector,x,y,vx,vy,k,Ny,Nx,Np,Nt,v)
            govector  = speedcoordinatecontrol(x,y,vx,vy,k,1,Nx,Ny)
        #----------
        #врахуємо процеси розсіювання:
        dispersal = whichdispersal(vx,vy,Np,Nt,m,k)
        x,y,vx,vy = diffusion_convert(dispersal, x, y, v, vx, vy, m, Np,k, Ny)
        #----------
    particles_on_each_itteration[voltagecounter] = sum(particlecounter)
    voltagecounter = voltagecounter + 1


import matplotlib.pyplot as plt
plt.plot(-Voltages, np.array(particles_on_each_itteration)*M*1.6e-19/dt*1000)
plt.xlabel('Voltage,V')
plt.ylabel('Current, mA')
plt.grid(True)

