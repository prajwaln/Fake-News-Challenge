import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version


def init_features(stances,dataset):
    h, b, y = [],[],[]

    for stance in stances:
        if stance['Predict'] != '?': continue
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    return h, b, y

def generate_features_all(stances,dataset,name):
    # Pass all articles through here first
    for stance in stances:
        stance['Stance'] = 'unrelated' if stance['Stance'] == 'unrelated' else 'discuss'
    h, b, y = init_features(stances,dataset)

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_overlap, X_hand]
    return X,y

def generate_features_related(stances,dataset,name):
    # Pass related articles through here second
    for stance in stances:
        stance['Stance'] = 'discuss' if stance['Stance'] == 'discuss' else 'disagree'
    h, b, y = init_features(stances,dataset)

    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_polarity, X_refuting]
    return X,y

def generate_features_partial(stances,dataset,name):
    # Pass partial articles through here third
    h, b, y = init_features(stances,dataset)

    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_polarity, X_refuting]
    return X,y

def run_stage(fn, folds, fold_stances, hold_out_stances, assignclasses):
    # Load/Precompute all features now
    Xs = dict()
    ys = dict()
    X_comp,y_comp = fn(competition_dataset.stances,competition_dataset,"competition")
    X_holdout,y_holdout = fn(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = fn(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    #Run on Holdout set and report the final score on the holdout set
    predicted_test = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual_test = [LABELS[int(a)] for a in y_holdout]
    assign_test_ids = [i for i in range(len(y_holdout)) if predicted_test[i] in assignclasses]
    for i in assign_test_ids: d.stances[i]['Predict'] = predicted_test[i]

    #Run on competition dataset
    predicted_comp = [LABELS[int(a)] for a in best_fold.predict(X_comp)]
    actual_comp = [LABELS[int(a)] for a in y_comp]
    assign_comp_ids = [i for i in range(len(y_comp)) if predicted_comp[i] in assignclasses]
    for i in assign_comp_ids: competition_dataset.stances[i]['Predict'] = predicted_comp[i]
    
    return (predicted_test, actual_test), (predicted_comp, actual_comp)

if __name__ == "__main__":
    check_version()
    parse_params()
    
    datapath = '../../'

    #Load the datasets and generate folds
    global d, competition_dataset
    d = DataSet(path=datapath)
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    competition_dataset = DataSet("competition_test", path=datapath)

    test, comp = run_stage(generate_features_all, folds, fold_stances, hold_out_stances, ['unrelated'])
    pred_test,actl_test = test
    pred_comp,actl_comp = comp
    
    # TODO: repeat stages for partial/impartial; agree/disagree

    print("Scores on the dev set")
    report_score(actl_test,pred_test)
    print("")
    print("")

    print("Scores on the test set")
    report_score(actl_comp,pred_comp)
