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
#    l_1 = max(l_1,eps)
#    l_2 = max(l_2,eps)
#    l_3 = max(l_3,eps)
#    x = min(values)
#    y = max(values)
#    t_0 = l_3
#    if l_1 < l_2:
#        t_1 = l_1
#        t_2 = l_2
#    else:
#        t_2 = l_1
#        t_1 = l_2
#
#    #p = np.zeros((x,y+1))
#    #p[0,y-x+1] = (np.exp(-t_1-t_2-t_0)/factorial(y-x+1))*t_2**(y-x+1)
#    #p[0,y-x] = (np.exp(-t_1-t_2-t_0)/factorial(y-x))*t_2**(y-x)
#    p_km_k = (np.exp(-t_1-t_2-t_0)/factorial(y-x+1))*t_2**(y-x+1)
#    p_km_km = (np.exp(-t_1-t_2-t_0)/factorial(y-x))*t_2**(y-x)
#
#    #for k in range(1,y-x+2):
#    #    p[0,k] = float(t_2)/k*p[0,k-1]
#    for k in range(1,x):
#        #p[k,y-x+k] = float(t_1)/k*p[k-1,y-x+k] +float(t_0)/k*p[k-1,y-x+k-1] 
#        #p[k,y-x+k+1] = float(t_2)/(y-x+k+1)*p[k,y-x+k] +float(t_0)/(y-x+k+1)*p[k-1,y-x+k] 
#        p_k_k = float(t_1)/k*p_km_k +float(t_0)/k*p_km_km 
#        p_k_kp = float(t_2)/(y-x+k+1)*p_k_k +float(t_0)/(y-x+k+1)*p_km_k
#        p_km_km = p_k_k
#        p_km_k = p_k_kp
#
#    return np.log(max(1e-9,float(t_1)/x*p_km_k+float(t_0)/x*p_km_km))
    
    #return -l_1-l_2-l_3+np.log(np.sum([(l_3**l/factorial(l)) * (l_1**(y_1-l)/factorial(y_1-l)) * (l_2**(y_2-l)/factorial(y_2-l)) for l in range(min(y_1,y_2)+1)]))
    #return pymc.poisson_like(y_1,l_1+l_3) + pymc.poisson_like(y_2,l_2+l_3)
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
#observed_matches = pd.read_csv('Prelim_RegularSeasonCompactResults_thru_Day132.csv')
#observed_matches = observed_matches[observed_matches['wteam'].isin(mm_teams) | observed_matches['lteam'].isin(mm_teams)][-640:].reset_index()
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

#observed data from games before the tournament
#N_obs = observed_matches.shape[0]
#observed_score = np.empty(N_obs, dtype=object)
#
#for n,match in observed_matches.iterrows():
#    fact = 1.+0.125*match.numot
#    if match.wloc == 'A':
#        hteam = team_to_ind[match.lteam]
#        ateam = team_to_ind[match.wteam]
#        hscore = int(match.lscore/fact)
#        ascore = int(match.wscore/fact)
#    else:
#        ateam = team_to_ind[match.lteam]
#        hteam = team_to_ind[match.wteam]
#        ascore = int(match.lscore/fact)
#        hscore = int(match.wscore/fact)
#    observed_score[n] = BivariatePoisson('observed_score_%i' % n,
#            l_1 = attack_strength[hteam]-defense_strength[ateam],
#            l_2 = attack_strength[ateam]-defense_strength[hteam],
#            l_3 = pace[hteam] + pace[ateam], observed = True, value = [hscore,ascore])
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

#odds_matches = []
#odds_winner = []
#odds_totals = []
#odds_spreads = []
#
##Dummy data
#for n in range(0,N,2):
#    odds_matches.append([n,n+1])
#    odds_spreads.append(np.random.randint(15))
#    odds_totals.append(100+np.random.randint(50))
#    odds_winner.append(0.5+0.5*np.random.random())
#
#home_score = np.empty(N/2, dtype=object)
#away_score = np.empty(N/2, dtype=object)
#pace_score = np.empty(N/2, dtype=object)
#home_score_pre = np.empty(N/2, dtype=object)
#away_score_pre = np.empty(N/2, dtype=object)
#total_score = np.empty(N/2, dtype=object)
#spread_score = np.empty(N/2, dtype=object)
#match_winner = np.empty(N/2, dtype=object)
#
#match_winner_potential = np.empty(N/2, dtype=object)
#total_score_potential = np.empty(N/2, dtype=object)
#spread_score_potential = np.empty(N/2, dtype=object)
#
#for match in range(len(odds_matches)):
#    hteam = odds_matches[match][0]
#    ateam = odds_matches[match][1]
#    home_score_pre[match] = pymc.Poisson('home_score_pre_%i' % match, 
#            mu = (attack_strength[hteam]-defense_strength[ateam] + abs(attack_strength[hteam]-defense_strength[ateam]))/2)
#    away_score_pre[match] = pymc.Poisson('away_score_pre_%i' % match, 
#            mu = (attack_strength[ateam]-defense_strength[hteam] + abs(attack_strength[ateam]-defense_strength[hteam]))/2)
#    pace_score[match] = pymc.Poisson('pace_score_%i' % match, mu = pace[ateam]+pace[hteam])
#    home_score[match] = home_score_pre[match] + pace_score[match]
#    away_score[match] = away_score_pre[match] + pace_score[match]
#    total_score[match] = home_score[match] + away_score[match]
#    spread_score[match] = home_score_pre[match] - away_score_pre[match]
#    match_winner[match] = home_score[match] > away_score[match]
#
#    home_score[match].__name__ = 'home_score_' + str(match)
#    away_score[match].__name__ = 'away_score_' + str(match)
#    total_score[match].__name__ = 'total_score_' + str(match)
#    spread_score[match].__name__ = 'spread_score_' + str(match)
#    match_winner[match].__name__ = 'match_winner_' + str(match)
#
#    home_score[match].keep_trace = True
#    away_score[match].keep_trace = True
#    total_score[match].keep_trace = True
#    spread_score[match].keep_trace = True
#    match_winner[match].keep_trace = True
#
#    #combine outcomes and odds
#    @pymc.potential
#    def match_win_pot(winner = match_winner[match], odds_win = odds_winner[match]):
#        if winner:
#            return np.log(odds_win)
#        else:
#            return np.log(1.-odds_win)
#    match_winner_potential[match] = match_win_pot
#
#    @pymc.potential
#    def total_score_pot(tot_sco = total_score[match], odds_tot = odds_totals[match]):
#        return pymc.distributions.poisson_like(tot_sco, mu=odds_tot)
#    total_score_potential[match] = total_score_pot
#
#    @pymc.potential
#    def spread_score_pot(spr_sco = spread_score[match], odds_spr = odds_spreads[match], odds_tot = odds_totals[match]):
#        return pymc.distributions.normal_like(spr_sco, mu=odds_spr, tau = 1./odds_tot)
#    spread_score_potential[match] = spread_score_pot
#
model = pymc.MCMC(locals())
model.sample(iter=37000, burn=1000, thin=10)
#pymc.Matplot.plot(model)
