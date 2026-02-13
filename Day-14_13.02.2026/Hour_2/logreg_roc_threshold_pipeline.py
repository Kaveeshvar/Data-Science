# logreg_roc_threshold_pipeline.py  # file name comment

import numpy as np  # import NumPy for numerical work
import pandas as pd  # import pandas for saving CSV artifacts
import matplotlib.pyplot as plt  # import matplotlib for plotting
from sklearn.datasets import make_classification  # import dataset generator
from sklearn.model_selection import train_test_split  # import split helper
from sklearn.linear_model import LogisticRegression  # import sklearn logistic regression
from sklearn.metrics import confusion_matrix  # import confusion matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # import metrics
from sklearn.metrics import roc_auc_score, roc_curve  # import ROC/AUC for verification


def make_data(seed=42):  # define function to generate a binary classification dataset
    X, y = make_classification(  # generate a synthetic dataset
        n_samples=5000,  # total number of samples
        n_features=20,  # number of features
        n_informative=8,  # how many features actually matter
        n_redundant=4,  # how many are redundant combinations
        n_clusters_per_class=2,  # clusters per class (harder than 1)
        weights=[0.7, 0.3],  # class imbalance (70% class0, 30% class1)
        flip_y=0.01,  # small label noise
        class_sep=1.0,  # separation between classes (lower => harder)
        random_state=seed,  # seed for reproducibility
    )  # end make_classification
    return X, y  # return features and labels


def confusion_at_threshold(y_true, y_score, t=0.5):  # compute confusion stats at threshold t
    y_pred = (y_score >= float(t)).astype(int)  # convert probabilities to 0/1 using threshold
    cm = confusion_matrix(y_true, y_pred)  # compute confusion matrix
    tn, fp, fn, tp = cm.ravel()  # unpack TN/FP/FN/TP for binary case
    acc = accuracy_score(y_true, y_pred)  # compute accuracy
    prec = precision_score(y_true, y_pred, zero_division=0)  # compute precision safely
    rec = recall_score(y_true, y_pred, zero_division=0)  # compute recall safely
    f1 = f1_score(y_true, y_pred, zero_division=0)  # compute F1 safely
    return tn, fp, fn, tp, acc, prec, rec, f1  # return metrics


def roc_from_scratch(y_true, y_score, thresholds):  # compute ROC points for a list of thresholds
    y_true = y_true.astype(int)  # ensure integer labels
    P = int(np.sum(y_true == 1))  # count positives
    N = int(np.sum(y_true == 0))  # count negatives

    tpr_list = []  # list to collect TPR values
    fpr_list = []  # list to collect FPR values

    for t in thresholds:  # loop over thresholds
        y_pred = (y_score >= float(t)).astype(int)  # threshold predictions
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # compute confusion components
        tpr = tp / (P + 1e-12)  # compute true positive rate safely
        fpr = fp / (N + 1e-12)  # compute false positive rate safely
        tpr_list.append(tpr)  # store TPR
        fpr_list.append(fpr)  # store FPR

    tpr_arr = np.array(tpr_list, dtype=np.float64)  # convert TPR list to array
    fpr_arr = np.array(fpr_list, dtype=np.float64)  # convert FPR list to array
    return fpr_arr, tpr_arr  # return ROC arrays


def auc_trapezoid(fpr, tpr):  # compute AUC using trapezoid rule
    order = np.argsort(fpr)  # sort by increasing FPR
    fpr_sorted = fpr[order]  # reorder FPR
    tpr_sorted = tpr[order]  # reorder TPR to match
    auc = np.trapezoid(tpr_sorted, fpr_sorted)  # compute area under curve via trapezoids
    return float(auc)  # return AUC


def threshold_sweep_metrics(y_true, y_score, thresholds, C_FP=1.0, C_FN=5.0):  # compute metrics per threshold
    rows = []  # list to store per-threshold metric dicts

    for t in thresholds:  # loop through thresholds
        y_pred = (y_score >= float(t)).astype(int)  # threshold predictions
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # compute confusion
        prec = precision_score(y_true, y_pred, zero_division=0)  # precision
        rec = recall_score(y_true, y_pred, zero_division=0)  # recall
        f1v = f1_score(y_true, y_pred, zero_division=0)  # F1
        P = (tp + fn)  # positives count
        N = (tn + fp)  # negatives count
        tpr = tp / (P + 1e-12)  # TPR
        fpr = fp / (N + 1e-12)  # FPR
        cost = float(C_FP) * float(fp) + float(C_FN) * float(fn)  # expected cost using counts

        rows.append(  # append row dict
            {  # begin dict
                "threshold": float(t),  # store threshold
                "tn": int(tn),  # store TN
                "fp": int(fp),  # store FP
                "fn": int(fn),  # store FN
                "tp": int(tp),  # store TP
                "precision": float(prec),  # store precision
                "recall": float(rec),  # store recall
                "f1": float(f1v),  # store F1
                "tpr": float(tpr),  # store TPR
                "fpr": float(fpr),  # store FPR
                "cost": float(cost),  # store cost
            }  # end dict
        )  # end append

    df = pd.DataFrame(rows)  # create DataFrame from rows
    return df  # return DataFrame


def pick_threshold_constraint(df, recall_target=0.95):  # pick threshold by constraint recall >= target, max precision
    feasible = df[df["recall"] >= float(recall_target)]  # filter thresholds meeting recall constraint
    if feasible.empty:  # if no threshold meets recall constraint
        return None  # return None to signal infeasible constraint
    best_idx = feasible["precision"].idxmax()  # choose row with maximum precision
    return feasible.loc[best_idx]  # return best row


def pick_threshold_cost(df):  # pick threshold minimizing cost
    best_idx = df["cost"].idxmin()  # index of minimum cost
    return df.loc[best_idx]  # return best row


def pick_threshold_f1(df):  # pick threshold maximizing F1
    best_idx = df["f1"].idxmax()  # index of max F1
    return df.loc[best_idx]  # return best row


def main():  # main driver function
    SEED = 42  # set seed constant
    np.random.seed(SEED)  # set NumPy global seed (extra reproducibility)

    # =========================
    # Hour 1 — Train + Scores
    # =========================

    X, y = make_data(seed=SEED)  # generate dataset
    y = y.astype(int)  # ensure y is int 0/1

    idx_all = np.arange(X.shape[0])  # create index array for reproducibility tracking

    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(  # split off test set
        X,  # features
        y,  # labels
        idx_all,  # indices to track rows
        test_size=0.20,  # 20% test
        random_state=SEED,  # seed
        stratify=y,  # stratified split (keeps class ratio)
    )  # end split

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(  # split temp into train+val
        X_temp,  # remaining features
        y_temp,  # remaining labels
        idx_temp,  # remaining indices
        test_size=0.25,  # 25% of 80% = 20% val
        random_state=SEED,  # seed
        stratify=y_temp,  # stratify again
    )  # end split

    # Save split indices for reproducibility (optional but recommended)
    np.save("idx_train.npy", idx_train)  # save train indices
    np.save("idx_val.npy", idx_val)  # save val indices
    np.save("idx_test.npy", idx_test)  # save test indices

    model = LogisticRegression(  # create sklearn logistic regression model
        penalty="l2",  # L2 regularization
        C=1.0,  # regularization strength inverse (bigger C => weaker reg)
        solver="lbfgs",  # robust solver for convex problems
        max_iter=2000,  # allow enough iterations to converge
    )  # end model init

    model.fit(X_train, y_train)  # fit model on training data

    y_score_val = model.predict_proba(X_val)[:, 1]  # get probability scores for class 1 on val
    y_score_test = model.predict_proba(X_test)[:, 1]  # get probability scores for class 1 on test

    # Baseline confusion matrix at threshold 0.5 (test set)
    tn, fp, fn, tp, acc, prec, rec, f1v = confusion_at_threshold(y_test, y_score_test, t=0.5)  # compute baseline
    print("\n[Hour 1] Baseline @ threshold=0.5 (TEST)")  # print header
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")  # print confusion counts
    print(f"acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1v:.4f}")  # print metrics

    # Save scores artifact (test)
    scores_df = pd.DataFrame(  # create scores DataFrame
        {  # begin dict
            "y_true": y_test.astype(int),  # store true labels
            "y_score": y_score_test.astype(np.float64),  # store predicted probabilities
        }  # end dict
    )  # end DataFrame
    scores_df.to_csv("scores.csv", index=False)  # save to CSV

    # =========================
    # Hour 2 — ROC + AUC
    # =========================

    thresholds = np.linspace(0.0, 1.0, 501)  # create threshold grid (0.002 steps)

    fpr_s, tpr_s = roc_from_scratch(y_test, y_score_test, thresholds)  # compute ROC curve from scratch
    auc_s = auc_trapezoid(fpr_s, tpr_s)  # compute AUC from scratch

    auc_sk = float(roc_auc_score(y_test, y_score_test))  # compute sklearn AUC
    fpr_sk, tpr_sk, _ = roc_curve(y_test, y_score_test)  # compute sklearn ROC curve

    print("\n[Hour 2] ROC/AUC verification (TEST)")  # print header
    print(f"AUC (scratch trapezoid) = {auc_s:.6f}")  # print scratch AUC
    print(f"AUC (sklearn)           = {auc_sk:.6f}")  # print sklearn AUC
    print(f"Abs diff                = {abs(auc_s - auc_sk):.6e}")  # print difference

    plt.figure()  # create figure
    plt.plot(fpr_s, tpr_s, label=f"Scratch ROC (AUC={auc_s:.4f})")  # plot scratch ROC
    plt.plot(fpr_sk, tpr_sk, linestyle="--", label=f"Sklearn ROC (AUC={auc_sk:.4f})")  # plot sklearn ROC
    plt.plot([0, 1], [0, 1], linestyle=":")  # plot random baseline diagonal
    plt.xlabel("False Positive Rate (FPR)")  # label x-axis
    plt.ylabel("True Positive Rate (TPR)")  # label y-axis
    plt.title("ROC Curve (Test Set)")  # title
    plt.legend()  # legend
    plt.tight_layout()  # adjust layout
    plt.savefig("roc.png", dpi=200)  # save ROC plot
    plt.show()  # show plot

    # =========================
    # Hour 3 — Threshold policy
    # =========================

    # Costs: tune these to your domain (security typically: FN cost >> FP cost)
    C_FP = 1.0  # cost of a false positive (e.g., extra manual review)
    C_FN = 10.0  # cost of a false negative (e.g., letting fraud/malware pass)

    sweep_df = threshold_sweep_metrics(  # compute metrics for each threshold
        y_test,  # true labels
        y_score_test,  # predicted probabilities
        thresholds,  # threshold grid
        C_FP=C_FP,  # FP cost
        C_FN=C_FN,  # FN cost
    )  # end sweep

    sweep_df.to_csv("threshold_sweep.csv", index=False)  # save threshold sweep to CSV

    # Policy A: constraint-based (recall >= target, maximize precision)
    # We must catch at least 95% of attacks
    # Find threshold where recall >= 0.95
    # Among those, pick the one with highest precision
    recall_target = 0.95  # example recall constraint
    best_constraint = pick_threshold_constraint(sweep_df, recall_target=recall_target)  # pick threshold by constraint

    print("\n[Hour 3] Threshold policy results (TEST)")  # header

    if best_constraint is None:  # if constraint infeasible
        print(f"Policy A (constraint): No threshold achieved recall >= {recall_target:.2f}")  # print
    else:  # if feasible
        print("Policy A (constraint): recall >= target, maximize precision")  # print description
        print(best_constraint[["threshold", "precision", "recall", "f1", "fpr", "tpr", "cost"]].to_string())  # print row

    # Policy B: cost-based (minimize expected cost)
    # Define:

    # C_FP = cost of false positive

    # C_FN = cost of false negative

    # Example:

    # C_FP = $1 (manual review cost)

    # C_FN = $100 (fraud loss)
    best_cost = pick_threshold_cost(sweep_df)  # pick threshold by cost
    print("\nPolicy B (cost): minimize C_FP*FP + C_FN*FN")  # print description
    print(f"C_FP={C_FP} C_FN={C_FN}")  # print costs
    print(best_cost[["threshold", "precision", "recall", "f1", "fpr", "tpr", "cost"]].to_string())  # print row

    # Policy C: F1-max (maximize F1)
    best_f1 = pick_threshold_f1(sweep_df)  # pick threshold by F1
    print("\nPolicy C (F1): maximize F1")  # print description
    print(best_f1[["threshold", "precision", "recall", "f1", "fpr", "tpr", "cost"]].to_string())  # print row

    # Security implications (short but real)
    print("\nSecurity implications (plain English):")  # print header
    print("- Lower threshold => more positives flagged => higher recall, higher FPR (more false alarms).")  # explain
    print("- Higher threshold => fewer positives flagged => lower FPR, but higher FN (missed attacks).")  # explain
    print("- In security, FN is often costlier (missed fraud/malware), so thresholds often bias toward recall.")  # explain
    print("- BUT: too many FP burns ops teams (alert fatigue), which becomes a security risk by itself.")  # explain

    print("\nArtifacts saved: scores.csv, threshold_sweep.csv, roc.png, idx_train.npy, idx_val.npy, idx_test.npy")  # print


if __name__ == "__main__":  # python entry point guard
    main()  # run main
