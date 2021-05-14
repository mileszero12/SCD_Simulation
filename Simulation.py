#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#import networkx as nx

#create state space and initial state prob
#states = ['NHD', 'HD', 'D']
#pi = [0.65, 0.35, 0] # initial prob



# In[6]:


# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

def createTransitionMatrix(p1, p2):
    q_df = pd.DataFrame(columns=states, index=states)
    q_df.loc[states[0]] = [1-p1, p1, 0]
    q_df.loc[states[1]] = [0, 1-p2, p2]
    q_df.loc[states[2]] = [0, 0, 1]

    return q_df

def createLMH(d):
    q_df = {}
    q_df['low'] = createTransitionMatrix(d['low'][0], d['low'][1])
    # med risk
    q_df['med'] = createTransitionMatrix(d['med'][0], d['med'][1])
    # high risk
    q_df['high'] = createTransitionMatrix(d['high'][0], d['high'][1])
    return q_df

    
def skipTransition(matrix, riskLevel, times):
    res = matrix[riskLevel]
    for i in range(times-2):
        res = res.dot(matrix[riskLevel])
    return res
    


# ***
# #### * $p_{tn}, p_{tp}$ can be changed.
# #### * If A is medum risk and the result of ECG is normal (2 yrs), then A's state will may change in 2 years.
# #### * Probability of quiting is dependent on riskLevel and current result.

# In[92]:


#cost = [200, 16, 5]
# consult , ECG , HP 

#QuitProb = {} # Y, N
#QuitProb['low'] = [[0.01, 0.99], [0.4, 0.6], [0.7, 0.3]]
#QuitProb['med'] = [[0.02, 0.98], [0.45, 0.55], [0.75, 0.25]]
#QuitProb['high'] = [[0.03, 0.97], [0.5, 0.5], [0.55, 0.45], [0.9, 0.1]]

#riskRate = {}
#riskRate['low'] = [0.1, 0.05] # p1, p2
#riskRate['med'] = [0.3, 0.12]
#riskRate['high'] = [0.4, 0.25]

#Accu = {} # different stragye may have different accuracy
#Accu['PTN'] = {'hp': 0.8, 'ecg': 0.9} # Healthy people correctly identified as healthy
#Accu['PTP'] = {'hp': 0.9, 'ecg': 0.99} # Sick people correctly identified as sick


# In[103]:

def CalulateCur(riskLevel, q_df, cur_age, cur_state, year):
    if year > 1:
        tmp = skipTransition(q_df, riskLevel, year)
    else: 
        tmp = q_df[riskLevel]
    n = cur_age + year
    next_state = np.random.choice(['NHD', 'HD', 'D'], p = tmp.loc[cur_state])
    return n, next_state

# flag = 0(both of the results are negative)
# flag = 1(hp's result is positive, and ecg's result is negative)
# flag = 2(hp's result is negative, and ecg's result is positive)
# flag = 3(both of the results are positive)
def decideQuit(riskLevel, flag, quitProb):
    if riskLevel == 'high':
        if flag == 0:
            return np.random.choice(['Y', 'N'], p = quitProb['high'][0])
        elif flag == 1:
            return np.random.choice(['Y', 'N'], p = quitProb['high'][1])
        elif flag == 2:
            return np.random.choice(['Y', 'N'], p = quitProb['high'][2])
        else:
            return np.random.choice(['Y', 'N'], p = quitProb['high'][3])
    elif riskLevel == 'med':
        if flag == 0:
            return np.random.choice(['Y', 'N'], p = quitProb['med'][0])
        elif flag == 1:
            return np.random.choice(['Y', 'N'], p = quitProb['med'][1])
        elif flag == 3:
            return np.random.choice(['Y', 'N'], p = quitProb['med'][2])
    else:
        if flag == 0:
            return np.random.choice(['Y', 'N'], p = quitProb['low'][0])
        elif flag == 1:
            return np.random.choice(['Y', 'N'], p = quitProb['low'][1])
        elif flag == 3:
            return np.random.choice(['Y', 'N'], p = quitProb['low'][2])


def NHDState(Accu, money): 
    flag = 0
    year = 1
    money += cost[2] # first we need to do hp test
    ptn = Accu['PTN']['hp']
    temp = np.random.choice(['TN', 'FP'], p = [ptn, 1-ptn]) # false positive should be penalized(Act: n, Pre:y)
    if temp == 'TN':# if hp negative we can stop, and set year to 2
        year += 1
        st.write('-->Screening interval:', year)
        st.write('-->HP test only')
        st.write('-->NHD: hp test TN')
        st.write('-->Cost:', money, '\n')
    else: # hp positive
        money += cost[1] # next we need to do ecg test
        ptn1 = Accu['PTN']['ecg']
        temp1 = np.random.choice(['TN', 'FP'], p = [ptn1, 1-ptn1])
        if temp1 == 'TN': # if ecg negative we can stop, and set year to 2
            flag = 1
            year += 1
            st.write('-->Screening interval:', year)
            st.write('-->HP test and ECG Test')
            st.write('-->NHD: hp test FP(hp FP penalty), ecg test TN')
            st.write('-->Cost:', money, '\n')
        else: # if ecg positive, we need to do the cardiology consult and set year to 1
            flag = 3
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->NHD: hp test FP(hp FP penalty), ecg test FP(ecg FP penalty)')
            st.write('-->Cost:', money, '\n')
    return money, year, flag
        
def NHDStateH(Accu, money): 
    flag = 0
    year = 1
    money += cost[2] # first we need to do hp test
    ptn = Accu['PTN']['hp']
    temp = np.random.choice(['TN', 'FP'], p = [ptn, 1-ptn]) # false positive should be penalized(Act: n, Pre:y)
    money += cost[1] # ecg test
    ptn1 = Accu['PTN']['ecg']
    temp1 = np.random.choice(['TN', 'FP'], p = [ptn1, 1-ptn1])
    if temp == 'TN':# hp negative
        if temp1 == 'TN': # ecg negative
            st.write('-->Screening interval:', year)
            st.write('-->HP test and ECG test')
            st.write('-->NHD: hp test TN, ecg test TN')
            st.write('-->Cost:', money, '\n')
        else: # ecg positive, we need to do the cardiology consult and set year to 1
            flag = 2
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->NHD: hp test TN, ecg test FP(ecg FP penalty)')
            st.write('-->Cost:', money, '\n')
    else: # hp positive
        if temp1 == 'TN': # if ecg negative we can stop
            flag = 1
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->NHD: hp test FP(hp FP penalty), ecg test TN')
            st.write('-->Cost:', money, '\n')
        else: # if ecg positive, we need to do the cardiology consult and set year to 1
            flag = 3
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->NHD: hp test FP(hp FP penalty), ecg test FP(ecg FP penalty)')
            st.write('-->Cost:', money, '\n')
    return money, 1, flag
    
def HDState(Accu, money):
    flag = 0
    year = 1
    money += cost[2] # first we need to do hp test
    ptp = Accu['PTP']['hp']
    temp = np.random.choice(['TP', 'FN'], p = [ptp, 1-ptp]) # false nagative should be penalized(Act: y, Pre:n)
    if temp == 'TP': # hp positive   
        money += cost[1]
        ptp1 = Accu['PTP']['ecg']
        temp1 = np.random.choice(['TP', 'FP'], p = [ptp1, 1-ptp1])
        if temp1 == 'TP': # ecg positive
            flag = 3
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->HD: hp test TP, ecg test TP')
            st.write('-->Cost:', money, '\n')
        else:
            flag = 1
            year += 1
            st.write('-->Screening interval:', year)
            st.write('-->HP test and ECG test')
            st.write('-->HD: hp test TP, ecg test FN(ecg FN penalty)')
            st.write('-->Cost:', money, '\n')
    else: # hp negative, penalization
        flag = 0
        year += 1
        st.write('-->Screening interval:', year)
        st.write('-->HP test only')
        st.write('-->HD: hp test FN(hp FN penalty)')
        st.write('-->Cost:', money, '\n')
    return money, year, flag
    

def HDStateH(Accu, money):
    flag = 0
    year = 1
    money += cost[2] # first we need to do hp test
    ptp = Accu['PTP']['hp']
    temp = np.random.choice(['TP', 'FN'], p = [ptp, 1-ptp]) # false nagative should be penalized(Act: y, Pre:n)
    money += cost[1] # ecg test
    ptp1 = Accu['PTP']['ecg']
    temp1 = np.random.choice(['TP', 'FN'], p = [ptp1, 1-ptp1]) 
    if temp == 'TP': # hp positive
        if temp1 == 'TP': # ecg positive
            flag = 3
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->HD: hp test TP, ecg test TP')
            st.write('-->Cost:', money, '\n')
        else: # ecg negative
            flag = 1
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->HD: hp test TP, ecg test FN(ecg FN penalty)')
            st.write('-->Cost:', money, '\n')
    else: # hp negative, penalization
        if temp1 == 'TP': # ecg positive
            flag = 2
            money += cost[0] # cardiology consult cost
            st.write('-->Screening interval:', year)
            st.write('-->HP test, ECG test, and Cardiology Consult')
            st.write('-->HD: hp test FN(hp FN penalty), ecg test TP')
            st.write('-->Cost:', money, '\n')
        else: # ecg negative
            st.write('-->Screening interval:', year)
            st.write('-->HP test and ECG test')
            st.write('-->HD: hp test FN(hp FN penalty), ecg test FN(ecg FN penalty)')
            st.write('-->Cost:', money, '\n')
    return money, 1, flag


def transState(riskLevel, cur_state, q_df, cur_age, age_upper, money, Accu, quitProb):
    st.write('\n')
    st.write('-->Current State:', cur_state)
    st.write('-->Current Age:', cur_age)
    tmp = q_df[riskLevel]
    if cur_age > age_upper:
        st.write('-->Age Limit')
        st.write('-->Exit System \n')
        return money
    else:
        if cur_state == 'NHD':
            if riskLevel == 'high':
                money, year, flag = NHDStateH(Accu, money)
            else:
                money, year, flag = NHDState(Accu, money)
            
            
            res_quit = decideQuit(riskLevel, flag, quitProb)
            if res_quit == 'Y':
                st.write('-->Athlete decide to quit')
                return money
            next_age, next_state = CalulateCur(riskLevel, q_df, cur_age, cur_state, year)
            
            newmoney = transState(riskLevel, next_state, q_df, next_age, age_upper, money, Accu, quitProb)
        
        elif cur_state == 'HD':
            if riskLevel == 'high':
                money, year, flag = HDStateH(Accu, money)
            else:
                money, year, flag = HDState(Accu, money)
            
            res_quit = decideQuit(riskLevel, flag, quitProb)
            if res_quit == 'Y':
                st.write('-->Athlete decide to quit')
                return money
            next_age, next_state = CalulateCur(riskLevel, q_df, cur_age, cur_state, year)
            
            newmoney = transState(riskLevel, next_state, q_df, next_age, age_upper, money, Accu, quitProb)

        else:
            st.write('-->Current State:', 'Dead')
            st.write('-->Cost: penalty', 10000)
            st.write('-->Die \n')
            money += 10000
            return money
    return newmoney

    

def Transform(ini_level, ini_state, q_df, cur_age, max_age, Accu, quitProb):
    st.write('Initial State:', ini_state)
    st.write('-->Risk Level:', ini_level)
    st.write('-->Current Age:', cur_age, '\n')
    c = transState(ini_level, ini_state, q_df, cur_age, max_age, 0, Accu, quitProb)
    st.write('Total Cost:', c)
    return c



import streamlit as st

st.sidebar.title("SCA/SCD Simulation for Young Athlete")

st.sidebar.header("Variables:")
rl = st.sidebar.selectbox('Risk Level:', ('low', 'med', 'high'))
s = st.sidebar.selectbox('Current State:',('NHD', 'HD'))
#rl = st.sidebar.text_input('', value='med')
#s = st.sidebar.text_input('Current State:', value='NHD')
curage = st.sidebar.number_input('Current Age:', value=15)
maxage = st.sidebar.number_input('Maximal(Retired) Age:', value=30)


st.sidebar.header("Initial Probability of NHD and HD:")
NHD = st.sidebar.number_input('Initial Probability of NHD(No Heart Disease)', value=0.65)
HD = st.sidebar.number_input('Initial Probability of HD(Heart Disease)', value=0.35)
states = ['NHD', 'HD', 'D']
pi = [NHD, HD, 0] # initial prob
state_space = pd.Series(pi, index=states, name='states')

st.sidebar.header("Cost:")
consult = st.sidebar.number_input('Cost for Cardiology Consult', value=200)
ECG = st.sidebar.number_input('Cost for ECG Test', value=16)
HP = st.sidebar.number_input('Cost for HP test', value=5)
cost = [consult, ECG, HP]
# consult , ECG , HP 

st.sidebar.header("Quit Probability:")
st.sidebar.subheader("* Type 1: Both HP and ECG results are negative")
st.sidebar.subheader("* Type 2: The HP result is positive, and the ECG result is negative")
st.sidebar.subheader("* Type 3: The HP result is positive, and the ECG result is negative")
st.sidebar.subheader("* Type 4: Both HP and ECG results are positive")
low1 = st.sidebar.number_input('The Quit Probability for Low Risk Patient when her/his result is type 1', value=0.01)
low2 = st.sidebar.number_input('The Quit Probability for Low Risk Patient when her/his result is type 2', value=0.4)
low3 = st.sidebar.number_input('The Quit Probability for Low Risk Patient when her/his result is type 4', value=0.7)
med1 = st.sidebar.number_input('The Quit Probability for Medium Risk Patient when her/his result is type 1', value=0.02)
med2 = st.sidebar.number_input('The Quit Probability for Medium Risk Patient when her/his result is type 2', value=0.45)
med3 = st.sidebar.number_input('The Quit Probability for Medium Risk Patient when her/his result is type 4', value=0.75)
high1 = st.sidebar.number_input('The Quit Probability for High Risk Patient when her/his result is type 1', value=0.03)
high2 = st.sidebar.number_input('The Quit Probability for High Risk Patient when her/his result is type 2', value=0.5)
high3 = st.sidebar.number_input('The Quit Probability for High Risk Patient when her/his result is type 3', value=0.55)
high4 = st.sidebar.number_input('The Quit Probability for High Risk Patient when her/his result is type 4', value=0.9)
QuitProb = {} # Y, N
#QuitProb['low'] = [[0.01, 0.99], [0.4, 0.6], [0.7, 0.3]]
#QuitProb['med'] = [[0.02, 0.98], [0.45, 0.55], [0.75, 0.25]]
#QuitProb['high'] = [[0.03, 0.97], [0.5, 0.5], [0.55, 0.45], [0.9, 0.1]]
QuitProb['low'] = [[low1, 1-low1], [low2, 1-low2], [low3, 1-low3]]
QuitProb['med'] = [[med1, 1-med1], [med2, 1-med2], [med3, 1-med3]]
QuitProb['high'] = [[high1, 1-high1], [high2, 1-high2], [high3, 1-high3], [high4, 1-high4]]

st.sidebar.header("Transition Probability:")
ll = st.sidebar.number_input('The Probability that the low risk patient in NHD(No Heart Disease) state currently will be in the HD(Heart Disease) state next stage', value=0.1)
lh = st.sidebar.number_input('The Probability that the low risk patient in HD(Heart Disease) state currently will be in the D(Death) state next stage', value=0.05)
ml = st.sidebar.number_input('The Probability that the medium risk patient in NHD(No Heart Disease) state currently will be in the HD(Heart Disease) state next stage', value=0.3)
mh = st.sidebar.number_input('The Probability that the medium risk patient in HD(Heart Disease) state currently will be in the D(Death) state next stage', value=0.12)
hl = st.sidebar.number_input('The Probability that the high risk patient in NHD(No Heart Disease) state currently will be in the HD(Heart Disease) state next stage', value=0.4)
hh = st.sidebar.number_input('The Probability that the high risk patient in HD(Heart Disease) state currently will be in the D(Death) state next stage', value=0.25)
riskRate = {}
#riskRate['low'] = [0.1, 0.05] # p1, p2
#riskRate['med'] = [0.3, 0.12]
#riskRate['high'] = [0.4, 0.25]
riskRate['low'] = [ll, hl]
riskRate['med'] = [ml, mh]
riskRate['high'] = [hl, hh]

st.sidebar.header("True Negative and True Positive Rate:")
ptn_hp = st.sidebar.number_input('The True Negative Rate for HP test', value=0.8)
ptn_ecg = st.sidebar.number_input('The True Negative Rate for ECG test', value=0.9)
ptp_hp = st.sidebar.number_input('The True Positive Rate for HP test', value=0.9)
ptp_ecg = st.sidebar.number_input('The True Positive Rate for ECG test', value=0.99)
Accu = {} # different stragye may have different accuracy
#Accu['PTN'] = {'hp': 0.8, 'ecg': 0.9} # Healthy people correctly identified as healthy
#Accu['PTP'] = {'hp': 0.9, 'ecg': 0.99} # Sick people correctly identified as sick
Accu['PTN'] = {'hp': ptn_hp, 'ecg': ptn_ecg}
Accu['PTP'] = {'hp': ptp_hp, 'ecg': ptp_ecg}



states = ['NHD', 'HD', 'D']
state_space = pd.Series(pi, index=states, name='states')
q_df = createLMH(riskRate)
costMoney = Transform(rl, s, q_df, curage, maxage, Accu, QuitProb) 
st.write(costMoney)
#costMoney = Transform('high', 'NHD', q_df, 15, 30, Accu) 
# ini_level, ini_state, q_df, cur_age, max_age, Accu


# In[115]:


#costMoney = Transform('med', 'HD', q_df, 15, 30, Accu) 


# # Parameters
# 1. cost: the cost of cardiology consult, ECG and, hp tests.
# 
# 2. QuitProb: the probability of quit for different risk levels and current result. 
# * $P(riskLevel = high) < P(riskLevel = med) < P(riskLevel = low)$
# * $P(current result = 3) < P(current result = 2) <P(current result = 1) < P(current result = 0)$ (current result = 0 refers to both of the results are negative,  current result = 1 refers to hp's result is positive, and ecg's result is negative, current result = 2 refers to hp's result is negative, and ecg's result is positive, current result = 3 refers to both of the results are positive)
# 
# 
# 3. riskRate: p1 and p2 for different risk level.
# 
# 4. Accu: the value of ptn and ptp for hp and ECG.
# 
# 5. q_df: use riskRate to form a transition matrix.
# 
# 6. riskLevel: high, med, and low.
# 
# 7. cur_state: NHD, HD, and D.
# 
# 8. max_age: if the athlete is older than this max_age, s/he will be retired.
# 
# 9. pi: the initial probability of NHD and HD.
# 
# 
# # Functions
# 1. transState: use patient's risk level, current state, and current age to estimate the cost for rach state.
# 
# 2. NHDState,HDState: use the patient's current state and current test results to decide whether this athlete needs to take the further test or consult or not, and then estimate the interval screening and cost. (only for med and low level)
# 
# 3. NHDStateH,HDStateH: same as NHDState,HDState, but for high level.
# 
# 4. CalulateCur: use the interval screening, current age, and state's transition matrix to estimate the age and state in the next stage, if the interval screening is larger than 1, we need to use $P^2$.
# 
# 5. decideQuit: base on risk level and current result, the athlete may decide to quit the screening. 

# In[121]:


