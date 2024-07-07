import random as r
import matplotlib.pyplot as m
import numpy as np
import imageio.v2 as im
import os
from pathlib import Path
import copy

path = os.path.abspath(Path(__file__).parent)
#path = (path.split("\Simulation_GIF_Finale.py"))[0]
print(path)
colors = [
(0.0, 1.0, 1.0),(1.0, 0.49, 0.251), (0.0, 0.0, 1.0), (0.498, 1.0, 0.0), (1.0, 1.0, 0.0),
(0.373, 0.62, 0.627), (0.647, 0.165, 0.165), (0.871, 0.722, 0.529), (0.816, 0.125, 0.565),(0.863, 0.863, 0.863),
(1.0, 0.843, 0.0), (0.251, 0.878, 0.816), (0.0, 0.78, 0.549), (0.933, 0.51, 0.933), (0.98, 0.922, 0.843),
(0.541, 0.169, 0.886),(0.612, 0.4, 0.122), (1.0, 0.922, 0.804),  (0.929, 0.569, 0.129), (0.541, 0.212, 0.059), 
(0.541, 0.2, 0.141), (1.0, 0.38, 0.012), (1.0, 0.6, 0.071),(0.824, 0.412, 0.118), (0.239, 0.349, 0.671), 
(0.239, 0.569, 0.251), (0.502, 0.541, 0.529), (1.0, 0.498, 0.314), (0.392, 0.584, 0.929), (0.89, 0.812, 0.341),
(1.0, 0.894, 0.769), (0.863, 0.078, 0.235), (0.0, 0.933, 0.933), (0.722, 0.525, 0.043), (0.0, 0.392, 0.0),
(0.498, 1.0, 0.831),(0.741, 0.718, 0.42), (0.333, 0.42, 0.184), (1.0, 0.549,0.0), (0.6, 0.196, 0.8), 
(0.914, 0.588, 0.478), (0.561, 0.737, 0.561), (0.282, 0.239, 0.545), (0.184, 0.31, 0.31), (0.0, 0.808, 0.82), 
(0.58, 0.0, 0.827), (1.0, 0.078, 0.576), (0.0, 0.749, 1.0), (0.412, 0.412, 0.412), (0.118, 0.565, 1.0), 
(0.988, 0.902, 0.788), (0.0, 0.788, 0.341), (0.698, 0.133, 0.133),(0.133, 0.545, 0.133), (0.961, 0.961, 0.863),
(0.961, 0.871, 0.702),  (0.855, 0.647, 0.125), (0.0, 0.502, 0.0), (0.502, 0.502, 0.412), 
(0.678, 1.0, 0.184), (0.941, 1.0, 0.941), (1.0, 0.412, 0.706), (0.69, 0.09, 0.122), (0.804, 0.361, 0.361), 
(0.294, 0.0, 0.51), (1.0, 1.0, 0.941), (0.161, 0.141, 0.129), (0.941, 0.902, 0.549), (0.902, 0.902, 0.98), 
(1.0, 0.941, 0.961), (0.486, 0.988, 0.0), (1.0, 0.98, 0.804), (0.678, 0.847, 0.902), (0.941, 0.502, 0.502),
(0.878, 1.0, 1.0), (1.0, 0.925, 0.545), (0.98, 0.98, 0.824), (1.0, 0.714, 0.757), (1.0, 0.627, 0.478), 
(0.125, 0.698, 0.667), (0.529, 0.808, 0.98), (0.518, 0.439, 1.0), (0.467, 0.533, 0.6), (0.69, 0.769, 0.871), 
(1.0, 1.0, 0.878), (0.196, 0.804, 0.196), (0.98, 0.941, 0.902), (1.0, 0.0, 1.0), (0.012, 0.659, 0.62), 
(0.502, 0.0, 0.0), (0.729, 0.333, 0.827), (0.576, 0.439, 0.859), (0.235, 0.702, 0.443), (0.482, 0.408, 0.933), 
(0.0, 0.98, 0.604), (0.282, 0.82, 0.8), (0.78, 0.082, 0.522), (0.89, 0.659, 0.412), (0.098, 0.098, 0.439), 
(0.741, 0.988, 0.788), (0.961,1.0, 0.98), (1.0, 0.894, 0.882), (1.0, 0.894, 0.71), (1.0, 0.871, 0.678), 
(0.0,0.0, 0.502), (0.992, 0.961, 0.902), (0.502, 0.502, 0.0), (0.42, 0.557, 0.137), (1.0, 0.502, 0.0), 
(1.0, 0.271, 0.0), (0.855, 0.439, 0.839), (0.933, 0.91, 0.667), (0.596, 0.984, 0.596), (0.733, 1.0, 1.0), 
(0.859, 0.439, 0.576), (1.0, 0.937,0.835), (1.0, 0.855, 0.725), (0.2, 0.631, 0.788), (1.0, 0.753, 0.796), 
(0.867, 0.627, 0.867), (0.69, 0.878, 0.902), (0.502, 0.0, 0.502), (0.529, 0.149, 0.341),(0.78, 0.38, 0.078), 
(1.0, 0.0, 0.0), (0.737, 0.561, 0.561), (0.255, 0.412, 0.882), (0.98, 0.502, 0.447), (0.957,0.643, 0.376), 
(0.188, 0.502, 0.078), (0.329,1.0, 0.624), (1.0, 0.961, 0.933), (0.369, 0.149, 0.071), (0.557, 0.22, 0.557), 
(0.773, 0.757, 0.667), (0.443, 0.776, 0.443), (0.49, 0.62, 0.753), (0.557, 0.557, 0.22), (0.776, 0.443, 0.443), 
(0.443, 0.443, 0.776), (0.22, 0.557, 0.557), (0.627, 0.322, 0.176), (0.753, 0.753, 0.753), (0.529, 0.808, 0.922),
(0.416, 0.353,0.804), (0.439, 0.502, 0.565), (1.0, 0.98, 0.98), (0.0, 1.0, 0.498), (0.275, 0.51, 0.706), 
(0.824, 0.706, 0.549), (0.0, 0.502, 0.502), (0.847, 0.749, 0.847), (1.0, 0.388, 0.278)
]
def table_alea (n,N,t,b) :
    '''
    Permet de créer une table aléatoirement suivant certains paramètres
    ------------------------------------------------
    n       : nombre de piles
    N       : nombre de cartes
    t       : nombre de type de cartes différent
    b       : taille minimum de pile
    '''
    quantitable = []
    proportable = []

    #QUANTITABLE
    N = N - b*n
    borne =sorted(r.sample(range(0,N),n-1))
    borne.append(N)
    temp = 0
    for j in range(0,n) :
        quantitable.append(borne[j] - temp +b)
        temp = borne[j]

    #PROPORTABLE
    for i in range(0,n):
        pile = []
        pourcent = sorted([r.randrange(0,quantitable[i]+1) for k in range(0,t-1)])
        pourcent.append(quantitable[i])
        temp=0
        for j in range(0,t) :
            pile.append((pourcent[j] - temp)/quantitable[i])
            temp = pourcent[j]
        proportable.append(pile)
    Q,P=(np.array(quantitable)).reshape(n,1),np.array(proportable)
    return Q,P


def simulation (n,N,t,b,e,R):
    '''

    ------------------------------------------------
    n       : nombre de piles
    N       : nombre de cartes
    t       : nombre de type de cartes différent
    b       : taille minimum de pile
    e       : nombre d'étapes
    R       : matrice de relation
    '''
    Q,P = table_alea(n,N,t,b)
    historique = [[copy.deepcopy(Q),copy.deepcopy(P)]]
    C = np.ones((n,1))
    L = np.ones((1,t))

    for k in range(0,e) :     
        P_=copy.deepcopy(P)
        Q_=copy.deepcopy(Q)
        S = np.zeros((n,t))
        P_2 = np.zeros((n,t))
        
        for i in range(0,n):
            for j in range(0,t):
                s=0
                for k in range(0,n):
                    s+= R[k][i]*P[k][j]
                S[i][j]+=s

        Q= copy.deepcopy(Q_ - (R @ C)+ (R.T @ C))

        for i in range(0,n):
            for j in range(0,t):
                P_2[i][j] += 1/(Q[i])
        
        P= copy.deepcopy(P_2*(P_ * ((Q_ - (R @ C))@ L) + S))

        historique.append([copy.deepcopy(Q),copy.deepcopy(P)])
    return historique



def afficher_histo (n,t,etat,name,save,show,e):
    '''
    n       : nombre de piles
    t       : nombre de type de cartes différent
    état    : état de la table au départ
    save    : Sauvegarder ou non l'histogramme
    show    : Montrer ou non l'histogramme
    e       : nombre d'étapes
    '''
    P = copy.deepcopy(etat[1])
    Q = copy.deepcopy(etat[0])
    max = np.max(Q)
    c = colors
    x=(np.linspace(1,n,n))
    
    L = np.ones((1,t))
    E= copy.deepcopy((P*(Q@L)).T)
    
    #m.grid()
    m.bar(x,[max for x in P],color='white',alpha=0.0) #Fond pour la stabilité
    m.figure(facecolor='#F6F5EC')
    B = np.zeros((1,n))
    for j in range(0,t):
        m.bar(x,E[j],bottom=B[0],color=c[j]) #Types suivants
        
        B = copy.deepcopy(B + E[j])
    
    m.title("Etape "+str(e))
    m.xticks(x,x)
  
    if save :   
        m.savefig(path+'/Gif/'+name)
    if show:
        m.show()
    m.clf()



def build_gif (n,t,historique) :
    '''    
    n           : nombre de piles
    t           : nombre de type de cartes différent
    historique  : List des états de la table pour chaque étape
    '''
    Lim = []
    for j in range(0,len(historique)):
        print('gif'+str(j))
        afficher_histo(n,t,historique[j],'gif'+str(j),True,False,j)
        Lim.append('gif'+str(j))

    with im.get_writer(path+'/Gif/'+'mygif.gif',mode='I',fps=2) as writer :
        for filename in Lim :
            image = im.imread(path+'/Gif/'+filename + '.png')
            writer.append_data(image)


r3 =  np.array([
[5,0,4],
[4,0,4],
[0,8,0]])

r8 = np.array([
[0,1,0,4,10,5,2,0],
[1,0,5,2,0,9,5,0],
[1,2,0,0,10,1,2,6],
[0,7,0,3,0,0,2,10],
[1,5,0,9,2,2,0,3],
[10,4,7,0,0,0,1,0],
[8,1,4,4,0,5,0,0],
[1,2,6,0,0,0,10,3]
])

r4 =  10*np.array([
[0,5,0,0],
[5,0,0,0],
[0,0,0,6],
[0,0,6,0]])

r2 =  np.array([[0,1],[1,1]])

#build_gif(3,4,simulation(3,50,4,10,10,r3))
build_gif(2,4,simulation(2,100,4,30,25,r2))
#build_gif(8,4,simulation(8,500,4,25,25,r8))
#build_gif(4,2,simulation(4,1000,2,100,25,r4))
