import random as r
import matplotlib.pyplot as m
import numpy as np




def conversion (table) :
    '''
    Transforme une table réelle en table de quantité
    -------------------------------------------------
    table : Table réelle
    '''
    n = len(table)
    l=[]
    for i in range (0,n):
        m = len(table[i])
        c = 0
        for j in range(0,m):
            if table[i][j] :
                c += 1
        l.append([(m-c) + c*1j])
    T1 = np.array(l)
    return T1

def proba (M) :
    '''
    Transforme une table de quantité en table de proportion
    -------------------------------------------------
    matrice : matrice de relation entre les piles
    '''
    I = M.imag
    R = M.real
    
    P = I/(I+R)
    return P

def next_etape (T1,R,C) :
    '''
    Passage de récurence de la table des cartes.
    -------------------------------------------------
    T1 : matrice des cartes à l'étape courante
    R : matrice de relations
    C : matrice colonne
    '''
    P = proba(T1)
    T2 = T1 - np.dot(R,C)*((-1+ 1j)*P+C) + np.dot(R.T,((-1+ 1j)*P+C))
    
    return T2

def verif_equilibre_matrice (matrice):
    '''
    Permet de vérifier si une matrice est équilibrée
    -------------------------------------------------
    matrice : matrice de relation entre les piles
    '''
    
    res = True
    for i in range(0,len(matrice)):
        si=0
        sj=0
        for j in range (0,len(matrice)):
            si = si+ matrice[i][j]
            sj = sj+ matrice[j][i]
        if sj != si :
            res= False
    return res

def execution_echange(T1,R,n,e):
    '''
    Echange les cartes, calcul la moyenne et affiche.
    -------------------------------------------------
    T1 : Table initiale (réelle)
    R : Matrice de relation entre les piles
    n : Nombre de piles
    e : Nombre d'étape
    '''


    l =[]
    for i in range(0,n):
        l.append([1])
    C= np.array(l)
    etape = [0]
    
    Prop = proba(T1)
    proportion_totale = (proba(np.dot((T1.T),C)))[0][0]
    print("proportion_totale : ",proportion_totale)

    liste_proportions_reduites = (Prop/proportion_totale).tolist() 
    
    T2=T1
    
    for j in range(e):
        T2 = next_etape(T2,R,C) 
        Prop2 = (((proba(T2)/proportion_totale).T)[0]).tolist()
        etape.append(j+1)
        for i in range(0,len(liste_proportions_reduites)):
            liste_proportions_reduites[i].append(Prop2[i])
    
    

    moy = []
    for e in range (0,len(liste_proportions_reduites[0])):
        temp = 0
        for k in range (0,len(liste_proportions_reduites)):
            temp += liste_proportions_reduites[k][e]
        moy.append(temp)
    
    moy = np.array(moy)
    moy = moy/len(liste_proportions_reduites)
    

    for j in range (0,len(liste_proportions_reduites)):
        m.bar(etape,liste_proportions_reduites[j],alpha=0.25)  
    
    
    m.xlabel("Echange n°")
    m.ylabel("Proportion de la pile / Proportion totale")
    m.title("Répartition des piles")
    
    legende = []
    for i in range(0,n):
        legende.append ("Pile n°"+str(i))
    m.legend(legende)
    m.plot(etape,moy,label="Moyenne à l'étape")
    
    m.show()

 

T1= conversion ([[True, True, True, True, False, False, False], [False, False, False, True, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False,False, False, False], [False, False, False, False, True, True, False, True, False, False, False, True, False, True, True, True, False, True, True, False, False, False, True, False, False, False,True, False, True], [False, False, False, True, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False,False, False, False]])

'''
R_ = [
[0,0,0,0,0,0,0,0],
[0,0,1,2,0,0,1,2],
[0,0,0,1,0,0,0,1],
[0,3,0,0,0,3,0,0],
[0,0,0,0,0,0,0,0],
[0,0,1,2,0,0,1,2],
[0,0,0,1,0,0,0,1],
[0,3,0,0,0,3,0,0]]
print(verif_equilibre_matrice(R_))
'''
'''
R_ = [
[0,0,0,3,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0]]
print(verif_equilibre_matrice(R_))

R_ = [
[0,0,0,0,1,0],
[0,0,1,2,0,1],
[0,0,0,1,1,0],
[0,3,0,0,0,1],
[1,0,1,0,1,0],
[0,1,0,1,0,1]]
print(verif_equilibre_matrice(R_))
'''

'''
R=np.array([
[0,0,0,0],
[0,0,1,2],
[0,0,0,1],
[0,3,0,0]])
'''
r8_ = [
[0,1,0,4,10,5,2,0],
[1,0,5,2,0,9,5,0],
[1,2,0,0,10,1,2,6],
[0,7,0,3,0,0,2,10],
[1,5,0,9,2,2,0,3],
[10,4,7,0,0,0,1,0],
[8,1,4,4,0,5,0,0],
[1,2,6,0,0,0,10,3]
]


r2_ = [[0,1],[1,1]]



R_ = np.array([
[0,3,0,0],
[0,0,7,0],
[0,0,0,0],
[0,2,0,0]])

P=proba(T1)
R = np.array(R_)

#execution_echange(T1,R,4,500)

r8_ = [
[0,1,0,4,10,5,2,0],
[1,0,5,2,0,9,5,0],
[1,2,0,0,10,1,2,6],
[0,7,0,3,0,0,2,10],
[1,5,0,9,2,2,0,3],
[10,4,7,0,0,0,1,0],
[8,1,4,4,0,5,0,0],
[1,2,6,0,0,0,10,3]
]

r2_ = [[0,1],[1,1]]

table8 = [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False], 
[False, False, False, True, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False,False, False, False], 
[False, False, False, False, True, True, False, True, False, False, False, True, False, True, True, True, False, True, True, False, False, False, True, False, False, False,True, False, True],
[True, True, False, False, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True,False, False, True], 
[True, True, False, True, False, False, True, False, True, False, False, True, True, False, False, False, False, False, False, True, False, False, True, True, False],
[ False,True, True, True, True, True, True, True, True, True, True, True, True, False,True, True, False, True, False, True, True, True, False, False, False, True, True, False, True, True, False, True, True],
[False, True, True, False, False, False, False, True, False, True,True, False, False, True, False, True, False, True, True, False, True, True, False, False, False, False, True, True, True, True, True, True, False, True, False, False, True, True, False, True,False, True, False, True, False, True, True, True, False, False],
[ True, True, True, False, True, False, False, True, True, False, True, False, False, False, True, False, False, True, True, True,False, True, True, True, False, False, True, True, False, True, False, False, True, True, False, False, False, False, True, True, True, True, False, True, False, True, False, True, True, True,True,True]]

table2 = [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False], 
[False, False, False, True, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False,False, False, False]]

r2 = np.array(r2_)
r8 = np.array(r8_)
T2= conversion (table2)
T8= conversion (table8)


#execution_echange(T2,r2,2,100)
execution_echange(T8,r8,8,10)

