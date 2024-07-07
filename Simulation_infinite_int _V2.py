import random as r
import matplotlib.pyplot as m
import numpy as np
import math as math
import imageio.v2 as im
import os
from pathlib import Path
import copy
import timeit
import time
import graphviz

path = os.path.abspath(Path(__file__).parent)
print(path)
colors = [
(0.0, 1.0, 1.0),(1.0, 0.49, 0.251), (0.0, 0.0, 1.0), (0.498, 1.0, 0.0), (1.0, 1.0, 0.0),
(0.373, 0.62, 0.627), (0.647, 0.165, 0.165), (0.871, 0.722, 0.529), (0.816, 0.125, 0.565),(0.863, 0.863, 0.863),
(1.0, 0.843, 0.0), (0.251, 0.878, 0.816), (0.0, 0.78, 0.549), (0.933, 0.51, 0.933), (0.98, 0.922, 0.843),
(0.541, 0.169, 0.886),(0.612, 0.4, 0.122), (1.0, 0.922, 0.804),  (0.929, 0.569, 0.129), (0.541, 0.212, 0.059),
(0.541, 0.2, 0.141), (1.0, 0.38, 0.012), (1.0, 0.6, 0.071),(0.824, 0.412, 0.118), (0.239, 0.349, 0.671),
(0.239, 0.569, 0.251), (0.502, 0.541, 0.529), (1.0, 0.498, 0.314), (0.392, 0.584, 0.929), (0.89, 0.812, 0.341)]


#######################################################################
########                         Matrices                ##############
#######################################################################

def prod_mat(A,B):
    C = []
    for i in range(0,len(A)):
        Cl=[]
        for j in range(0,len(A)):
            s = 0
            for k in range(0,len(A)):
                s += A[i][k]*B[k][j]
            Cl.append(s)
        C.append(Cl)
    return C

def scal_prod_mat(a,A):
    C=A
    for i  in range(0,len(A)):
        for j in range(0,len(A[0])): 
            C[i][j]= a*A[i][j]
    return C

def scal_div_mat(a,A):
    C=A
    for i  in range(0,len(A)):
        for j in range(0,len(A)): 
            C[i][j]= A[i][j]/a
    return C

def pwr_mat(Q,n):
    C = copy.deepcopy(Q)
    for i in range(0,n-1):
        C = prod_mat(C,Q)
    return C

def dif_mat(A,B):
    C=[]
    for i in range(0,len(A)):
        Cl=[]
        for j in range(0,len(A[0])):
            Cl.append(A[i][j]-B[i][j])
        C.append(Cl)
    return C

def dif_list(A,B):
    C=[]
    for i in range(0,len(A)):
        C.append(A[i]-B[i])
    return C

def abs_list(A):
    C=[]
    for i in range(0,len(A)):
        C.append(abs(A[i]))
    return C

def sum_list(A,B):
    C = []
    for i in range(len(A)):
        C.append(A[i]+B[i])
    return C

def scal_prod_list(a,L):
    C=[]
    for i in range(0,len(L)):
        C.append(a*L[i])
    return C

def scal_div_list(a,L):
    C=[]
    for i in range(0,len(L)):
        l = L[i]
        C.append(l/a)
    return C

def sum_pile(X):
    N= []
    for i in range(0,len(X)):
        s=0
        for j in range(0,len(X[0])):
            s += X[i][j]
        N.append(s)
    return N   

def max_pile(X):
    N=sum_pile(X)
    m=0
    for i in range(0,len(N)):
        if N[i]>m :
            m = N[i]
    return m

def transpose(X):
    p = len(X)
    t = len(X[0])
    C =[]
    for j in range(0,t):
        Cl = []
        for i in range(0,p):
            Cl.append(X[i][j])
        C.append(Cl)
    return C

    
def piles_repartition(X):
    N=sum_pile(X)
    Y=[]
    p = len(X)
    t = len(X[0])

    for i in range(0,p):
        Yl = []
        for j in range(0,t):
            Yl.append(X[i][j]/N[i])
        Y.append(Yl)
    return Y

def sum_coef_list(L):
    s=0
    for i in range(0,len(L)):
        s += L[i]
    return s

def min_list(L):
    min = L[0]
    for e in L:
        if e<min:
            min = e
    return min

#######################################################################
########                         Etapes                  ##############
#######################################################################

def transforme_R_Q (R,X0,p,t):
    N = []
    for i in range(0,p):
        s=0
        for j in range(0,t):
            s+=X0[i][j]
        N.append(s)
    
    Nmax = 1
    for i in range(0,p):
        Nmax = Nmax*N[i]
    print(Nmax)

    Q=[]
    for i in range(0,p):
        Ql=[]
        for j in range(0,p):
            if i==j :
                s = 0
                for k in range(0,p):
                    if k!=i:
                        s+= R[j][k]
                l = int(int(Nmax)-int(s*(Nmax/N[i])))
                Ql.append(l)
            else :
                Ql.append(int((R[i][j])*int(Nmax/N[i])))
        Q.append(Ql)

    return Q,Nmax

def etape(X,Q,n,Nmax):
    Qn = scal_div_mat(int(Nmax**n),pwr_mat(Q,n))
    Xn=[]
    for j in range(0,len(X)):
        s = scal_prod_list(Qn[0][j],X[0])
        for i in range(1, len(X)):
            s = sum_list(s,scal_prod_list(Qn[i][j],X[i]))
        Xn.append(s)
    return Xn

def etape_succ(X,Q,n,Nmax):
    H=[X]
    for e in range (1,n+1):
        H.append(etape(X,Q,e,Nmax))        
    return H

#######################################################################
########                         Affichage               ##############
#######################################################################

def mono_image(nom,X,affiche):
    Y = transpose(X)
    pile = []
    maxX=[]
    base = []

    max = 1.1*max_pile(X)
    p = len(X)
    t = len(X[0])

    for i in range(0,p):
        pile.append(i)
        maxX.append(max)
        base.append(0)
    
    m.bar(pile,maxX,color='black',alpha=0.0)

    for j in range(0,t):
        m.title(nom)
        m.bar(pile,Y[j],bottom=base,color=colors[j],alpha=1.0)
        base = sum_list(Y[j],base)
    
    m.xticks(pile,pile)
    m.savefig(path+'/Gif/'+nom+".png")
    if affiche :
        m.show()
    m.clf()
    return path+'/Gif/'+nom+".png"

def gif (nom,X,Q,n,affiche,Nmax):
    H = etape_succ(X,Q,n,Nmax)
    i = 0
    Lim=[]
    for Xn in H :
        Lim.append(mono_image(nom+str(i),Xn,affiche))
        i+=1
    
    with im.get_writer(path+'/Gif/'+nom+'.gif',mode='I',fps=5) as writer :
        for filename in Lim :
            image = im.imread(filename)
            writer.append_data(image)

#######################################################################
########                  Apprentissage                  ##############
#######################################################################

def fusionne_fam(X,i,j):
    X0 =[]
    for k in range(0,len(X)):
        if k == i : 
            X0.append(  [X[i][0] + X[j][0]]+ X[i][1:(len(X[i]))] + X[j][1:(len(X[j]))]   )
        elif k != j : 
            X0.append(X[k])    
    print("Fus : ",i,j)  
    print(X)
    print(X0)      
    return X0

def distance_vec (x,y) :
    d = abs_list(dif_list(x,y))
    s=0
    for i in range(0,len(x)):
        s += d[i]*d[i]
    return np.sqrt(s)


def distance_fam_min (U,V):
    #Incompatible avec c_h_a car ne prend pas les numéros de piles
    min = distance_vec(U[0],V[0])
    for e in U :
        for f in V:
            d = distance_vec(e,f)
            if d<min :
                min = d
    return min

def distance_fam_moy (U,V): 
    #Incompatible avec c_h_a car ne prend pas les numéros de piles
    s = 0
    for e in U :
        for f in V:
            s += distance_vec(e,f)
    return s/(len(U)*len(V))

def barycentre_fam(U):
    s= copy.deepcopy(U[1])
    for i in range(2,len(U)):
        s = sum_list(s,U[i])
    return scal_div_list((len(U)-1),s)

def distance_fam_Ward (U,V):
    bu = barycentre_fam(U)
    bv = barycentre_fam(V)
    return (np.sqrt(((len(U)-1)*(len(V)-1))/(len(U)+len(V)-2)))*distance_vec(bu,bv)

def classification_hierarchique_ascendante(X,M):
    X0=[]
    
    for k in range(0,len(X)):
        X0.append([[k],tuple(scal_div_list(sum_coef_list(X[k]),X[k]))])

    print(X0)
    d_f_min=0
    while len(X0)>1 and d_f_min<M:
        d_f_min = distance_fam_Ward(X0[0],X0[1])
        pair_min = (0,1)
        for i in range(0,len(X0)):
            for j in range(i+1,len(X0)):
                if distance_fam_Ward(X0[i],X0[j]) < d_f_min:
                    d_f_min = distance_fam_Ward(X0[i],X0[j])
                    pair_min = (i,j)
        if d_f_min<M :
            X0 = fusionne_fam(X0,pair_min[0],pair_min[1])
    
    return X0


#######################################################################
########                         Graphe                  ##############
#######################################################################
        
        
def graphe_fam(R,F):
    dot = graphviz.Digraph('Table',format='png')  
    for k in range(0,len(F)):
        c = graphviz.Digraph(name="cluster_"+str(k))
        c.attr(label="Famille "+str(k))
        for l in F[k]:
            c.node(str(l))
        dot.subgraph(c)
    """
    for j in range(0,len(R)):
        for i in range(0,len(R)):
            if R[i][j] != 0 :
                dot.edge(str(i),str(j),label=str(R[i][j]),fontcolor='lightgrey')"""
    dot.render(directory='doctest-output', view=True)  



def parcours_profondeur_postordre (R):
    etat =[]
    parent=[]
    debut=[]
    fin=[]
    postordre =[]
    for i in range(0,len(R)):
        etat.append(0)
        parent.append(i)
        debut.append(-1)
        fin.append(-1)

    temps=0

    def visiter (R,k,etat,parent,debut,fin,postordre,temps):
 

        (etat2,parent2,debut2,fin2,postordre2,temps2) = (etat,parent,debut,fin,postordre,temps+1)
        debut2[k]=temps2
        etat2[k]=1
        for j in range(0,len(R)):
            if R[k][j]>0 and etat2[j]==0:
                etat2[j]=k
                (etat2,parent2,debut2,fin2,postordre2,temps2)= visiter(R,j,etat2,parent,debut,fin,postordre,temps+1)
        temps2 +=1
        fin2[k]=temps2
        postordre2.append(k)
        etat2[k]=2
        return (etat2,parent2,debut2,fin2,postordre2,temps2)
    
    for i in range(0,len(R)):
        if etat[i]==0 :
            (etat,parent,debut,fin,postordre,temps)=visiter(R,i,etat,parent,debut,fin,postordre,temps)
    return postordre

def parcours_profondeur_cpst_frtmt_cnx(R,L):
    etat =[]
    composantes_fortement_connexes =[]
    f=-1
    
    for i in range(0,len(R)):
        etat.append(0)

    temps=0

    def visiter (R,k,etat,composantes_fortement_connexes,f,temps):

        (etat2,composantes_fortement_connexes2,f2,temps2)=(etat,composantes_fortement_connexes,f,temps)
        temps2 +=1
        etat2[k]=1
        for j in range(0,len(R)):
            if R[k][j]>0 and etat2[j]==0:
                etat2[j]=k
                (etat2,composantes_fortement_connexes2,f2,temps2) = visiter(R,j,etat2,composantes_fortement_connexes2,f2,temps2)
        temps2 +=1
        etat2[k]=2
        composantes_fortement_connexes2[f2].append(k)
        return (etat2,composantes_fortement_connexes2,f2,temps2)
    print(L)
    for s in L:
        if etat[s]==0 :
            f+=1
            composantes_fortement_connexes.append([])
            (etat,composantes_fortement_connexes,f,temps)=visiter(R,s,etat,composantes_fortement_connexes,f,temps)
    return composantes_fortement_connexes



def kosaraju(R):
    postordre_parcours = parcours_profondeur_postordre(R)
    print(postordre_parcours)

    composantes_fortements_connexes = parcours_profondeur_cpst_frtmt_cnx(transpose(R),list(reversed(postordre_parcours)))

    return composantes_fortements_connexes

def transforme_FamVect_Fam(X0):
    F = []
    for i in range(0,len(X0)):
        F.append(X0[i][0])
    return F

def trie_Famille(F):

    F_min=[]
    for k in range(0,len(F)):
        F_min.append(min_list(F[k]))

    max =0
    for i in range(0,len(F_min)):
            if F_min[i]>max:
                max = F_min[i]

    list_min=[]

    for j in range(0,len(F_min)):
        min=F_min[0]
        imin=0
        for i in range(0,len(F_min)):
            if F_min[i]<min:
                imin=i
                min = F_min[i]
        list_min.append(imin)
        F_min[imin]=max+1
    Fam=[]
    for k in range(0,len(list_min)):
        Fam.append(sorted(F[list_min[k]]))

    return(Fam)
        
def test_equilibre(R,X,p,t,n,M):
    equilibre =[]
    K=trie_Famille(kosaraju(R))
    Q,Nmax=transforme_R_Q(R,X,p,t)
    for i in range(1,n+1):
        Xi=etape(X,Q,i,Nmax)
        Fi=trie_Famille(transforme_FamVect_Fam(classification_hierarchique_ascendante(Xi,M)))
        equilibre.append(K==Fi)
    return equilibre
        
def rang_equilibre(T):
    equ = []
    nequ=[]
    for b in range(0,len(T)):
        if T[b] :
            equ.append(b)
        else :
            nequ.append(b)
    
    if nequ[-1]<len(T)-1 :
        return (True,(nequ[-1])+1)
    else :
        (False,len(T))

    




#######################################################################
########                         Appels                  ##############
#######################################################################


r4 =  [
[0,5,0,0],
[5,0,0,0],
[0,0,0,6],
[0,0,6,0]]
r4 = scal_prod_mat(2,r4)

r8 = [
[0,1,0,4,10,5,2,0],
[1,0,5,2,0,9,5,0],
[1,2,0,0,10,1,2,6],
[0,7,0,3,0,0,2,10],
[1,5,0,9,2,2,0,3],
[10,4,7,0,0,0,1,0],
[8,1,4,4,0,5,0,0],
[1,2,6,0,0,0,10,3]
]

r17 = [
[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
'''
X17 = [[1,2],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3],[0,3]]
Q17,Nmax17=transforme_R_Q(r17,X17,17,2)
gif("R17-Etape-",X17,Q17,200,False,Nmax17)
#T = test_equilibre(r17,X17,17,2,200,0.01)
#(b17,e17) = rang_equilibre(T)
#print(str(b17)," -> ",e17)'''

'''X8 = [[20,0,10,10],[20,10,30,50],[20,20,30,30],[30,30,30,0],[20,30,20,10],[0,20,0,0],[30,10,30,0],[20,10,0,30]]
Q8,Nmax8 = transforme_R_Q(r8,X8,8,4)
F8=classification_hierarchique_ascendante(etape(X8,Q8,5,Nmax8),0.01)
'''
K8 = kosaraju(r8)
#print(test_equilibre(r8,X8,8,3,50,0.001))
#print(trie_Famille(K8)==trie_Famille(transforme_FamVect_Fam(F8)))
#graphe_fam(r8,trie_Famille(transforme_FamVect_Fam(F8)))
#gif("R8-Etape-",X8,Q8,100,False,Nmax8)
graphe_fam(r8,trie_Famille(K8))

"""
K4 = kosaraju(r4)
graphe_fam(r4,trie_Famille(K4))"""
'''
X4 = [[20,30,10],[10,10,20],[30,0,10],[20,20,10]]
X4 = scal_prod_mat(10,X4)
Q4,Nmax4 = transforme_R_Q(r4,X4,4,3)
#F4 = classification_hierarchique_ascendante(etape(X4,Q4,50,Nmax4),0.1)

#graphe_fam(r4,F4)
#gif("R4-Etape-",X4,Q4,100,False,Nmax4)
#K4 = kosaraju(r4)
#print(trie_Famille(K4)==trie_Famille(transforme_FamVect_Fam(F4)))

r2=[[0,1],[1,1]]
X2 = [[20,30,10,0],[30,0,10,20]]
Q2,Nmax2 = transforme_R_Q(r2,X2,2,4)
#gif("R2-Etape-",X2,Q2,25,False,Nmax2)'''


