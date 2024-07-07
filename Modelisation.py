import random as r
import matplotlib.pyplot as m

def echange_bilateral_kitaire (l1,l2,k):
    '''
    l1 : pile numéro 1
    l2 : pile numéro 2
    k : Nombre carte à échanger
    '''
    
    E12 = l1[0:k]
    E21 = l2[0:k]
    l1 = E21 + l1[k:len(l1)]
    l2 = E12 + l2[k:len(l2)]
    return [l1,l2]
    
def melange_pile (l):
    '''
    l : pile à mélanger
    '''
    r.shuffle(l)


def affichage (table):
    '''
    table : table à afficher
    '''
    ligne = ""
    for i in table:
        for j in i:
            if j:
                ligne = ligne + "○"
            elif not j:
                ligne = ligne +"■"
        ligne = ligne + "  "
    
    print(ligne)

def melange_table (table):
    '''
    table : table à mélanger
    '''
    for i in range(0, len(table)):
        melange_pile(table[i])

def initialise_table (n_min,n_max,n_pile):
    '''
    n_min : Nombre minimum de cartes dans une pile
    n_max : Nombre maximum de cartes dans une pile
    n_pile : nombre de piles
    '''
    table = []
    for p in range(0,n_pile):
        longueur = r.randint(n_min, n_max)
        cibles = r.randint(0,longueur)
        pile =[]
        if cibles != 0:
            for j in range(0,cibles):
                pile.append(True)
        if cibles != longueur:
            for j in range(cibles,longueur):
                pile.append(False)
        table.append(pile)
    melange_table(table)
    print("Table base   " ,end='')
    affichage(table)
    return table
    
def initialise_table_2 (n_min,n_max,n_pile,n_carte):
    '''
    n_min : Nombre minimum de cartes dans une pile
    n_max : Nombre maximum de cartes dans une pile
    n_pile : Nombre de piles
    n_carte : Nombre totale de cartes (> n_min * n_pile)
    '''
    table = []
    
    for p in range(0,n_pile -1) :
        longueur = r.randint(n_min, min(n_max,n_carte - (n_pile -1 -p)))
        n_carte = n_carte - longueur
        cibles = r.randint(0,longueur)
        pile =[]    
        if cibles != 0:
            for j in range(0,cibles):
                pile.append(True)
        if cibles != longueur:
            for j in range(cibles,longueur):
                pile.append(False)
        table.append(pile)
    longueur = n_carte
    cibles = r.randint(0,longueur)
    pile =[]
    if cibles != 0:
        for j in range(0,cibles):
            pile.append(True)
    if cibles != longueur:
        for j in range(cibles,longueur):
            pile.append(False)
    table.append(pile)
    melange_table(table)
    print("Table base   " ,end='')
    affichage(table)
    return table
    
    
def compte_pile (l):
    '''
    l : pile à dénombrer
    '''
    compteur =0
    for i in range(0,len(l)):
        if l[i] :
            compteur =compteur + 1
    return compteur

def compte_table (table):
    '''
    table : table à dénombrer
    '''
    compteur = []
    for i in range(0,len(table)):
        compteur.append(compte_pile(table[i]))
    return compteur



def experience_2_piles(n_echange,n_min,n_max,taille_paquet) :
    '''
    n_echange : nombre d'echange
    n_min : Nombre minimum de cartes dans une pile
    n_max : Nombre maximum de cartes dans une pile
    taille_paquet : Nombre de cartes échangées par échange
    '''
    
    table = initialise_table (n_min,n_max,2)
    compteur = compte_table(table)
    repartition_tot = (compteur[0] + compteur[1])/(len(table[0]) + len(table[1]))
    
    #Data graphique
    x = [0]
    y1 = [(compteur[0]/len(table[0])) /repartition_tot]
    y2 = [(compteur[1]/len(table[1])) /repartition_tot]
    
    #Exécution des échanges
    for i in range(0,n_echange):
        table = echange_bilateral_kitaire(table[0],table[1],taille_paquet)
        compteur = compte_table(table)
        #print(repartition_tot)
        
        melange_table(table)
        x.append(i+1)
        y1.append((compteur[0]/len(table[0])) /repartition_tot)
        y2.append((compteur[1]/len(table[1])) /repartition_tot)
        print("Echange n°",i+1,"  ",y1[i],"/",y2[i],"   ",end='')
        
        #affichage(table)
    
    #Graphique
    m.bar(x,y1,alpha=0.25)
    m.bar(x,y2,alpha=0.25)
    m.hlines([1],[-0.66],[n_echange+0.66])
    
    m.xlabel("Echange n°")
    m.ylabel("Proportion de la pile / Proportion totale")
    m.title("Répartition des piles")
    m.text(0,0.05,"Taille pile n°1 : "+str(len(table[0])) + " \nTaille pile n°2 : " +str(len(table[1]))+ " \n Proportion globale : "+ str(repartition_tot))
    m.show()

def echange_circulaire_kitaire (table,k):
    '''
    table : Liste des piles à échanger
    k : Nombre carte à échanger
    '''
    paquets = []
    
    for i in range(0,len(table)):
        paquets.append(table[i][0:k])
    for i in range(0,len(table)):
        table[i] = paquets[i-1] + table[i][k:len(table[i])]
    
    return table
    
def moyenne (l) :
    moy = []
    for j in range (0,len(l[0])):
        s=0
        for i in range(0,len(l)):
            s = s + l[i][j]
        moy.append(s/len(l))
    return moy

def experience_n_piles(n,n_echange,n_min,n_max,taille_paquet):
    '''
    n : Nombre de piles
    n_echange : nombre d'echange
    n_min : Nombre minimum de cartes dans une pile
    n_max : Nombre maximum de cartes dans une pile
    taille_paquet : Nombre de cartes échangées par échange
    '''
    
    table = initialise_table (n_min,n_max,n)
    compteur = compte_table(table)
    compteur_tot = 0
    cartes_tot = 0
    
    for i in range(0,n):
        compteur_tot += compteur[i]
        cartes_tot += len(table[i])

    repartition_tot = compteur_tot/cartes_tot
    
    #Data graphique
    x = [0]
    y=[]
    moy=[]
    
    for i in range(0,n):
        y.append([(compteur[i]/len(table[i])) /repartition_tot])
    
    moy.append(moyenne(y))
    #Execution
    for i in range(0,n_echange) :
        table = echange_circulaire_kitaire(table,taille_paquet)
        compteur = compte_table(table)
        melange_table(table)
        
        x.append(i+1)
        for j in range(0,n):
            y[j].append((compteur[j]/len(table[j])) /repartition_tot)
        moy.append(moyenne(y))
    
    for i in range(0,n):
        
        m.bar(x,y[i],alpha=0.25)
        
    m.plot(x,moy)
    #m.hlines([1],[-0.66],[n_echange+0.66])

    m.plot(x,moy)
    m.xlabel("Echange n°")
    m.ylabel("Proportion de la pile / Proportion totale")
    m.title("Répartition des piles")
    
    m.show()

def echange_matriciel(table,matrice):
    '''
    table : Liste de piles à échanger
    matrice : matrice de relation entre les piles
    Exemple : 
    [[1,2,3,0,2,0],
     [0,0,0,0,1,0],
     [1,2,3,0,0,3],
     [0,0,4,0,1,0],
     [0,1,0,0,0,0],
     [0,0,0,0,1,2]]
    '''
    
    paquets = []

    for i in range(0,len(matrice)):
        paquets.append([])
        for j in range(0,len(matrice)) :
            k = matrice[i][j]
            p=table[i][0:k]
            paquets[i].append(p)
            table[i] = table[i][k:len(table[i])]
    #print (table)
    #print(paquets)
    for i in range(0,len(matrice)):
        for j in range(0,len(matrice)) :
            table[j] = table[j]+paquets[i][j]
    
    #print (table)
    return table




def experience_matricielle(n,n_echange,n_min,n_max,matrice):
    '''
    n : Nombre de piles
    n_echange : nombre d'echange
    n_min : Nombre minimum de cartes dans une pile
    n_max : Nombre maximum de cartes dans une pile
    matrice : matrice de relation entre les piles
    '''
    
    table = initialise_table (n_min,n_max,n)
    compteur = compte_table(table)
    compteur_tot = 0
    cartes_tot = 0
    
    for i in range(0,n):
        compteur_tot += compteur[i]
        cartes_tot += len(table[i])

    repartition_tot = compteur_tot/cartes_tot
    
    
    legende = []
    for i in range(0,n):
        legende.append ("Pile n°"+str(i)+" : "+str(compteur[i])+"/"+str(len(table[i])))
    
    
    
    #Data graphique
    x = [0]
    y=[]
    moy=[]
    
    for i in range(0,n):
        
        y.append([(compteur[i]/len(table[i])) /repartition_tot])
    
    moy.append(moyenne(y))
    #Execution
    for i in range(0,n_echange) :
        table = echange_matriciel(table,matrice)
        #affichage(table)
        compteur = compte_table(table)
        melange_table(table)
        
        x.append(i+1)
        for j in range(0,n):
            if len(table[j]) != 0 :
                y[j].append((compteur[j]/len(table[j])) /repartition_tot)
            else:
                y[j].append(0)
            
        moy.append(moyenne(y))
  
    
    for i in range(0,n):
        legende[i] = legende[i] + " -> Pile n°"+str(i)+" : "+str(compteur[i])+"/"+str(len(table[i]))
        m.bar(x,y[i],alpha=0.25)
    
    print(legende)
    m.legend(legende)
    m.plot(x,moy)
    #m.hlines([1],[-0.66],[n_echange+0.66])
    
    m.xlabel("Echange n°")
    m.ylabel("Proportion de la pile / Proportion totale")
    m.title("Répartition des piles")
    
    m.show()


def experience_matricielle_2 (table,n_pile,n_echange,matrice,show):
    '''
    table : Table de départ
    n_pile : Nombre de piles
    n_echange : nombre d'echange
    n_min : Nombre minimum de cartes dans une pile
    n_max : Nombre maximum de cartes dans une pile
    n_carte : Nombre totale de cartes (> n_min * n_pile)
    matrice : matrice de relation entre les piles
    '''
    
    
    compteur = compte_table(table)
    compteur_tot = 0
    cartes_tot = 0
    
    for i in range(0,n_pile):
        compteur_tot += compteur[i]
        cartes_tot += len(table[i])

    repartition_tot = compteur_tot/cartes_tot
    
    
    legende = []
    for i in range(0,n_pile):
        legende.append ("Pile n°"+str(i)+" : "+str(compteur[i])+"/"+str(len(table[i])))
    
    
    
    #Data graphique
    x = [0]
    y=[]
    moy=[]
    
    for i in range(0,n_pile):
        y.append([(compteur[i]/len(table[i])) /repartition_tot])
    
    #Execution
    for i in range(0,n_echange) :
        table = echange_matriciel(table,matrice)
        #affichage(table)
        compteur = compte_table(table)
        melange_table(table)
        
        x.append(i+1)
        for j in range(0,n_pile):
            if len(table[j]) != 0 :
                y[j].append((compteur[j]/len(table[j])) /repartition_tot)
            else:
                y[j].append(0)
            
    
    moy=moyenne(y)
    
    
    if show :
        for i in range(0,n_pile):
            legende[i] = legende[i] + " -> Pile n°"+str(i)+" : "+str(compteur[i])+"/"+str(len(table[i]))
            m.bar(x,y[i],alpha=0.25)
        
        
        m.legend(legende)
        m.plot(x,moy,label="Moyenne à l'étape")
        m.hlines([1],[-0.66],[n_echange+0.66],label='Equilibre global')
        
        m.xlabel("Echange n°")
        m.ylabel("Proportion de la pile / Proportion totale")
        m.title("Répartition des piles")
        
        m.axis(xmin=0, ymin=0)
        m.show()

    return moy

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

def etude_experiences_matricielles(table,matrice,n_pile,n_echange,n_test):
    '''
    Permet de représenter les moyennes de répartition des pile à chaque étape pour une série d'expérience. 
    Permet de visualiser une moyenne des moyennes.
    ------------------------------------------------
    table : Table de départ
    matrice : matrice de relation entre les piles
    n_pile : Nombre de piles
    n_echange : nombre d'echange
    n_test : Nombre d'experiences successives
    '''
    etape = []
    for j in range(0,n_echange+1):
        etape.append(j)
    
    moy=[]
    for k in range(0,n_test):
        moy.append(experience_matricielle_2(table,n_pile,n_echange,matrice,False)) 
        
    moymoy = moyenne(moy)
    m.plot(etape,moymoy,linewidth=1,color='b')
    for i in range(0,len(moy)):
        m.plot(etape,moy[i],linewidth=0.2)
        
    m.legend(["Moyenne des moyennes"])
    m.xlabel("Etape n°")
    m.ylabel("Moyennes des experiences")
    
    m.axis(xmin=0, ymin=0)
    m.show()




r2 = [[0,1],[1,1]]


matrice66=[
[1,2,3,0,2,0],
[0,0,0,0,1,0],
[1,2,3,0,0,3],
[0,0,4,0,1,0],
[0,1,0,0,0,0],
[0,0,0,0,1,2]]

matrice44=[
[0,0,0,0],
[0,0,1,2],
[0,0,0,1],
[0,3,0,0]]

print("Vérification matrice équilibrée : ",verif_equilibre_matrice(matrice66))


#table = initialise_table_2 (5,50,4,300)
#print(table)

table = [[True, True, True, True, True, True, False], [False, False, False, True, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False,False, False, False], [False, False, False, False, True, True, False, True, False, False, False, True, False, True, True, True, False, True, True, False, False, False, True, False, False, False,True, False, True], [True, True, False, False, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True,False, False, True, True, True, False, True, False, False, True, False, True, False, False, True, True, False, False, False, False, False, False, True, False, False, True, True, False, False,False,True, True, False, True, False, True, True, True, False, False, False, True, True, False, True, True, False, True, True, False, True, True, False, False, False, False, True, False, True,True, False, False, True, False, True, False, True, True, False, True, True, False, False, False, False, True, True, True, True, True, True, False, True, False, False, True, True, False, True,False, True, False, True, False, True, True, True, False, False, True, True, True, False, True, False, False, True, True, False, True, False, False, False, True, False, False, True, True, True,False, True, True, True, False, False, True, True, False, True, False, False, True, True, False, False, False, False, True, True, True, True, False, True, False, True, False, True, True, True,True,True, True, True, True, True, True, False, False, True, True, True, True, True, False, True, False, True, True, True, False, True, False, False, True, True, False, False, False, False, True,True, False, False, False, True, True, True, False, False, True, True, False, True, True, False, False, True, True, False, False, False, True, True, False, True, True, False, True, False, True,False, True]]

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

table6 = [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False], 
[False, False, False, True, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False,False, False, False], 
[False, False, False, False, True, True, False, True, False, False, False, True, False, True, True, True, False, True, True, False, False, False, True, False, False, False,True, False, True],
[True, True, False, False, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True,False, False, True], 
[True, True, False, True, False, False, True, False, True, False, False, True, True, False, False, False, False, False, False, True, False, False, True, True, False],
[ False,True, True, True, True, True, True, True, True, True, True, True, True, False,True, True, False, True, False, True, True, True, False, False, False, True, True, False, True, True, False, True, True]]


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

#res = experience_matricielle_2(table8,8,200,r8,True)
#res = experience_matricielle_2(table8,6,200,matrice66,True)
#res = experience_matricielle_2(table,4,200,matrice44,True)
#etude_experiences_matricielles(table8,r8,8,200,50)

#res = experience_matricielle_2(table2,2,200,r2,True)
etude_experiences_matricielles(table2,r2,2,50,50)
    


#matrice22 =[ [0,2],[1,0]]
#experience_matricielle(2,10,10,10,matrice22)









#experience_2_piles(500,1,2000,1)









