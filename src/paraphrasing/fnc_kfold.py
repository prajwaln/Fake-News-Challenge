import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, \
                                word_overlap_features, naive_bayes_features, naive_bayes_train
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

runpass = 0

def init_features(stances,dataset,repl={}):
    id, h, b, y = [],[],[],[]

    for stance in stances:
        id.append(stance['Stance ID'])
        s = stance['Stance']
        y.append(LABELS.index(repl[s] if s in repl else s))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    
    # Train Naive Bayes
    print('Training Naive Bayes classifier...')
    naive_bayes_train(h, b, y)

    return id, h, b, y

def generate_features_all(stances,dataset,name):
    # Pass all articles through here first
    id, h, b, y = init_features(stances,dataset,{'agree':'discuss','disagree':'discuss'})

    X_bayes = gen_or_load_feats(naive_bayes_features, h, b, "features/bayes."+name+".npy")
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_overlap, X_hand, X_bayes]
    return X,y,id

def generate_features_related(stances,dataset,name):
    # Pass related articles through here second
    id, h, b, y = init_features(stances,dataset,{'agree':'disagree'})

    X_bayes = gen_or_load_feats(naive_bayes_features, h, b, "features/bayes."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_polarity, X_refuting, X_hand, X_bayes]
    return X,y,id

def generate_features_biased(stances,dataset,name):
    # Pass biased articles through here third
    id, h, b, y = init_features(stances,dataset)

    X_bayes = gen_or_load_feats(naive_bayes_features, h, b, "features/bayes."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_polarity, X_refuting, X_hand, X_bayes]
    return X,y,id

def run_stage(fn, d, competition_dataset):
    global runpass
    runpass += 1
    
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    
    # Load/Precompute all features now
    Xs = dict()
    ys = dict()
    ids = dict()
    comp_stances = competition_dataset.get_unlabelled_stances()
    X_comp,y_comp,id_comp = fn(comp_stances,competition_dataset,"competition_{}".format(str(runpass)))
    X_holdout,y_holdout,id_holdout = fn(hold_out_stances,d,"holdout_{}".format(str(runpass)))
    for fold in fold_stances:
        Xs[fold],ys[fold],ids[fold] = fn(fold_stances[fold],d,"{}_{}".format(str(fold),str(runpass)))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        id_train = np.hstack(tuple([ids[i] for i in range(len(fold_stances)) if i != fold]))
        X_train = np.vstack(tuple([Xs[i] for i in range(len(fold_stances)) if i != fold]))
        y_train = np.hstack(tuple([ys[i] for i in range(len(fold_stances)) if i != fold]))
        id_test = ids[fold]
        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted_test = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual_test = [LABELS[int(a)] for a in y_test]
        for i in range(len(actual_test)):
            d.stances[id_test[i]]['Predict'] = actual_test[i] # Data is known

        fold_score, _ = score_submission(actual_test, predicted_test)
        max_fold_score, _ = score_submission(actual_test, actual_test)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    #Run on Holdout set and report the final score on the holdout set
    predicted_hold = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual_hold = [LABELS[int(a)] for a in y_holdout]
    for i in range(len(predicted_hold)):
        d.stances[id_holdout[i]]['Predict'] = predicted_hold[i] # Data is unknown

    #Run on competition dataset
    predicted_comp = [LABELS[int(a)] for a in best_fold.predict(X_comp)]
    actual_comp = [LABELS[int(a)] for a in y_comp]
    for i in range(len(actual_comp)):
        competition_dataset.stances[id_comp[i]]['Predict'] = predicted_comp[i] # Data is unknown
    
    return id_holdout

def print_scores(test, comp, id):
    pred_test = [s["Predict"] for s in test.get_labelled_stances() if s["Stance ID"] in id]
    actl_test = [s["Stance"]  for s in test.get_labelled_stances() if s["Stance ID"] in id]
    pred_comp = [s["Predict"] for s in comp.get_labelled_stances()]
    actl_comp = [s["Stance"]  for s in comp.get_labelled_stances()]

    print("Scores on the dev set")
    report_score(actl_test,pred_test)
    print("")
    print("")

    print("Scores on the test set")
    report_score(actl_comp,pred_comp)

if __name__ == "__main__":
    check_version()
    parse_params()
    
    datapath = '../../'

    d = DataSet(path=datapath)
    competition_dataset = DataSet("competition_test", path=datapath)

    id = run_stage(generate_features_all, d, competition_dataset)
    print_scores(d, competition_dataset, id)

    # Clear placeholder values
    for s in d.stances:
        s['Predict'] = s['Predict'] if s['Stance']  == 'unrelated' else '?'
    for s in competition_dataset.stances:
        s['Predict'] = s['Predict'] if s['Predict'] == 'unrelated' else '?'

    id = run_stage(generate_features_related, d, competition_dataset)
    print_scores(d, competition_dataset, id)

    # Clear placeholder values
    for s in d.stances:
        s['Predict'] = s['Predict'] if s['Stance']  in ['discuss','unrelated'] else '?'
    for s in competition_dataset.stances:
        s['Predict'] = s['Predict'] if s['Predict'] in ['discuss','unrelated'] else '?'

    id = run_stage(generate_features_biased, d, competition_dataset)
    print_scores(d, competition_dataset, id)
