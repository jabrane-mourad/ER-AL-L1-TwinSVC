from sklearn.ensemble import RandomForestClassifier



N_INSTANCES =1;     P_INITI=30;      P_QUERIES=40;     NI = 10
ESTIMATOR = RandomForestClassifier(n_estimators=100, criterion='entropy');
DATA_BASE ="S_BeerAdvo_RateBeer"      ;       train= 268      ;           test= 91
# DATA_BASE ="S_iTunes_Amazon"          ;       train= 321      ;           test= 109
# DATA_BASE ="D_iTunes_Amazon"          ;       train= 321      ;           test= 109
# DATA_BASE ="S_Fodors_Zagats"          ;       train= 567      ;           test= 189
# DATA_BASE ="D_wdc_headphones"         ;       train= 1163     ;           test= 290
# DATA_BASE ="D_wdc_phones"             ;       train= 1762     ;           test= 440
# DATA_BASE ="T_abt_buy"                ;       train= 5743     ;           test= 1916

# DATA_BASE ="S_Walmart-Amazon"         ;       train= 6144     ;           test= 2049
# DATA_BASE ="D_Walmart_Amazon"         ;       train= 6144     ;           test= 2049
# DATA_BASE ="S_Amazon_Google"          ;       train= 6874     ;           test= 2293

# DATA_BASE ="T_Amazon_Google"          ;       train= 6753     ;           test= 1687
# DATA_BASE ="D_DBLP_ACM"               ;       train= 7417     ;           test= 2473
# DATA_BASE ="S_DBLP_ACM"               ;       train= 7417     ;           test= 2473
#DATA_BASE ="D_DBLP_GoogleScholar"     ;       train= 17223    ;           test= 5742
# DATA_BASE ="S_DBLP_GoogleScholar"     ;       train= 17223    ;           test= 5742
N_INITIAL = int(P_INITI*train/100);directory='with_clustring'; BOOTSTARP_STRATEGY = 'TwinSVMClustering'
N_QUERIES= int(P_QUERIES*train/100);

Exploration = 'Exploration';
Exploitation = 'Exploitation';
DPQ = 'DPQ';
STRATEGIES = [DPQ]
