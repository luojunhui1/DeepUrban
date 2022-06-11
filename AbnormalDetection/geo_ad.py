from .BaseSVDD import *
from numpy import *
from sklearn.ensemble import IsolationForest

def svdd_ad(data, thresh = None):
    svdd = BaseSVDD(C=thresh, gamma=0.3, kernel='rbf', display='off')
    svdd.fit(data)
    anomaly_score = svdd.get_distance(data) - svdd.radius
    anomaly_score = array(anomaly_score).squeeze()
    return where(anomaly_score > 0)[0]

def iforest_ad(data, thresh = None):
    clf = IsolationForest(contamination=0.02, max_features=2)
    clf.fit(data)
    anomaly_score = clf.score_samples(data)
    anomaly_score = (anomaly_score-min(anomaly_score))/(max(anomaly_score)-min(anomaly_score))
    anomaly_score = 1 - anomaly_score
    if thresh == None:
        thresh = 90
    return where(anomaly_score >= percentile(anomaly_score, thresh))[0]

def ksigma(flow, k = 3):
    m = mean(flow)
    std = np.std(flow, ddof=1)
    return where(abs(flow - m) > k*std)[0]

def chisquare_ad(data, k = 3):
    means = mean(data,axis=1)
    chi = [sum(((data[i] - means)**2)/means) for i in range(0, len(data))]
    cur = chi - mean(chi)
    anomaly_score = [0 if cur[i] < 0 else cur[i] for i in range(0, len(cur))]
    return where(anomaly_score >= k)[0]

