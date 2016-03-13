import numpy as np
import pymc

N = 68

attack_strength = np.empty(N, dtype=object)
defense_strength = np.empty(N, dtype=object)
pace = np.empty(N, dtype=object)

for i in range(N):
    attack_strength[i] = pymc.Exponential('attack_strength_%i' % i,0.01)
    defense_strength[i] = pymc.Exponential('defense_strength_%i' % i,0.01)
    pace[i] = pymc.Exponential('pace_%i' % i,0.1)

odds_matches = []
odds_winner = []
odds_totals = []
odds_spreads = []

#Dummy data
for n in range(0,N,2):
    odds_matches.append([n,n+1])
    odds_spreads.append(np.random.randint(15))
    odds_totals.append(100+np.random.randint(50))
    odds_winner.append(0.5+0.5*np.random.random())

home_score = np.empty(N/2, dtype=object)
away_score = np.empty(N/2, dtype=object)
pace_score = np.empty(N/2, dtype=object)
home_score_pre = np.empty(N/2, dtype=object)
away_score_pre = np.empty(N/2, dtype=object)
total_score = np.empty(N/2, dtype=object)
spread_score = np.empty(N/2, dtype=object)
match_winner = np.empty(N/2, dtype=object)

match_winner_potential = np.empty(N/2, dtype=object)
total_score_potential = np.empty(N/2, dtype=object)
spread_score_potential = np.empty(N/2, dtype=object)

for match in range(len(odds_matches)):
    hteam = odds_matches[match][0]
    ateam = odds_matches[match][1]
    home_score_pre[match] = pymc.Poisson('home_score_pre_%i' % match, 
            mu = (attack_strength[hteam]-defense_strength[ateam] + abs(attack_strength[hteam]-defense_strength[ateam]))/2)
    away_score_pre[match] = pymc.Poisson('away_score_pre_%i' % match, 
            mu = (attack_strength[ateam]-defense_strength[hteam] + abs(attack_strength[ateam]-defense_strength[hteam]))/2)
    pace_score[match] = pymc.Poisson('pace_score_%i' % match, mu = pace[ateam]+pace[hteam])
    home_score[match] = home_score_pre[match] + pace_score[match]
    away_score[match] = away_score_pre[match] + pace_score[match]
    total_score[match] = home_score[match] + away_score[match]
    spread_score[match] = home_score_pre[match] - away_score_pre[match]
    match_winner[match] = home_score[match] > away_score[match]

    home_score[match].__name__ = 'home_score_' + str(match)
    away_score[match].__name__ = 'away_score_' + str(match)
    total_score[match].__name__ = 'total_score_' + str(match)
    spread_score[match].__name__ = 'spread_score_' + str(match)
    match_winner[match].__name__ = 'match_winner_' + str(match)

    home_score[match].keep_trace = True
    away_score[match].keep_trace = True
    total_score[match].keep_trace = True
    spread_score[match].keep_trace = True
    match_winner[match].keep_trace = True

    #combine outcomes and odds
    @pymc.potential
    def match_win_pot(winner = match_winner[match], odds_win = odds_winner[match]):
        if winner:
            return np.log(odds_win)
        else:
            return np.log(1.-odds_win)
    match_winner_potential[match] = match_win_pot

    @pymc.potential
    def total_score_pot(tot_sco = total_score[match], odds_tot = odds_totals[match]):
        return pymc.distributions.poisson_like(tot_sco, mu=odds_tot)
    total_score_potential[match] = total_score_pot

    @pymc.potential
    def spread_score_pot(spr_sco = spread_score[match], odds_spr = odds_spreads[match], odds_tot = odds_totals[match]):
        return pymc.distributions.normal_like(spr_sco, mu=odds_spr, tau = 1./odds_tot)
    spread_score_potential[match] = spread_score_pot

model = pymc.MCMC(locals())
