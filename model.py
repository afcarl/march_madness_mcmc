import matplotlib
matplotlib.use('Agg')
import numpy as np
import pymc
import pandas as pd
from math import factorial

# http://www.economists.nl/files/20130411-SN2004.pdf
eps = 1e-3
import biv_pois
def bivariate_poisson_like_classic(values, l_1, l_2, l_3):
    y_1 = values[0]
    y_2 = values[1]
    return -l_1-l_2-l_3+np.log(np.sum([(l_3**l/factorial(l)) * (l_1**(y_1-l)/factorial(y_1-l)) * (l_2**(y_2-l)/factorial(y_2-l)) for l in range(min(y_1,y_2)+1)]))

def bivariate_poisson_like(values, l_1, l_2, l_3):
    return biv_pois.bivariate_poisson_like(values[0],values[1], l_1, l_2, l_3)

def rbivariate_poisson(l_1,l_2,l_3):
    l_1 = max(l_1,eps)
    l_2 = max(l_2,eps)
    l_3 = max(l_3,eps)
    x = pymc.rpoisson(l_3)
    return [pymc.rpoisson(l_1)+x,pymc.rpoisson(l_2)+x]

BivariatePoisson = pymc.stochastic_from_dist('BivariatePoisson', logp = bivariate_poisson_like, random = rbivariate_poisson, dtype=np.int, mv=True)

bracket = pd.read_csv("TourneySlots.csv")
mm_teams = pd.read_csv("TourneySeeds.csv")
seeds = dict(zip(mm_teams[mm_teams['Season']==2016]['Seed'],mm_teams[mm_teams['Season']==2016]['Team']))
mm_teams = mm_teams[mm_teams['Season']==2016]['Team'].unique()
teams = mm_teams #list(set(np.hstack((observed_matches['lteam'].unique() , observed_matches['wteam'].unique()))))

N = len(teams)
team_to_ind = dict(zip(teams,range(N)))

attack_strength = np.empty(N, dtype=object)
defense_strength = np.empty(N, dtype=object)
pace = np.empty(N, dtype=object)

for i in range(N):
    attack_strength[i] = pymc.Exponential('attack_strength_%i' % i,0.01)
    defense_strength[i] = pymc.Exponential('defense_strength_%i' % i,0.05)
    pace[i] = pymc.Exponential('pace_%i' % i,0.1)

opn = pd.read_csv('open_odds.csv')
win = pd.read_csv('win_odds.csv')
id_to_name = dict(zip(win['id'],win.name))
opn_dict = dict(zip(zip(opn.t_home,opn.t_away),(1./opn.o_home+1./opn.o_away)/opn.o_home))
opn_dict.update(dict(zip(zip(opn.t_away,opn.t_home),(1./opn.o_home+1./opn.o_away)/opn.o_away)))
win_dict = dict(zip(win.id,np.log(1/win.odds)))
predicted_score = np.empty(len(mm_teams)*(len(mm_teams)-1)/2, dtype=object)

i = 0
teams_to_match = dict()
match_winner_potential = []
for n,t1 in enumerate(mm_teams):
    for t2 in mm_teams[n+1:]:
        teams_to_match[(t1,t2)] = i
        teams_to_match[(t2,t1)] = i
        ateam = team_to_ind[t1]
        hteam = team_to_ind[t2]
        predicted_score[i] = BivariatePoisson('predicted_score_%i' % i,
            l_1 = attack_strength[hteam]-defense_strength[ateam],
            l_2 = attack_strength[ateam]-defense_strength[hteam],
            l_3 = pace[hteam] + pace[ateam])
        if t1 in id_to_name.keys() and t2 in id_to_name.keys() and (id_to_name[t1],id_to_name[t2]) in opn_dict.keys(): 
            print (id_to_name[t1],id_to_name[t2])
            @pymc.potential
            def match_win_pot(score = predicted_score[i], odds=opn_dict[(id_to_name[t1],id_to_name[t2])]):
                if score[0]>score[1]:
                    return np.log(odds)
                else:
                    return np.log(1.-odds)
            match_winner_potential.append(match_win_pot)
        i += 1
match_winner_potential = np.array(match_winner_potential)

@pymc.deterministic(trace=True)
def played_winner(predicted_score=predicted_score):
    played = np.zeros(predicted_score.shape[0]+1)*np.nan
    winner = seeds.copy()
    for _,m in bracket.iterrows():
        match_ind = teams_to_match[(winner[m.Strongseed],winner[m.Weakseed])]
        if predicted_score[match_ind][0]>predicted_score[match_ind][1]:
            winner[m.Slot] = winner[m.Strongseed]
            played[match_ind] = 1.
        else:
            winner[m.Slot] = winner[m.Weakseed]
            played[match_ind] = 0.
    played[-1] = winner['R6CH']
    return played

@pymc.potential
def tourney_winner_pot(played_winner=played_winner):
    return win_dict[played_winner[-1]]

model = pymc.MCMC(locals())
model.sample(iter=37000, burn=1000, thin=10)
#pymc.Matplot.plot(model)
