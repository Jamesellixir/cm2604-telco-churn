"""Evaluate frozen choices without tuning; export aggregate results only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from src.data import load_data, split_data, CSV_SHA256
from src.models import build_default_models, build_baseline

FROZEN_SHA256 = "a3c7a508f0777bbf24d88e8d2bdd2c9875904446bdab0e1398382c8a533addf5"

def metrics(labels, predictions, probability):
    """Binary churn metrics; absent positive predictions receive precision zero."""
    return dict(accuracy=accuracy_score(labels,predictions),
        balanced_accuracy=balanced_accuracy_score(labels,predictions),
        precision=precision_score(labels,predictions,zero_division=0),
        recall=recall_score(labels,predictions,zero_division=0),
        f1=f1_score(labels,predictions,zero_division=0),
        roc_auc=roc_auc_score(labels,probability),
        average_precision=average_precision_score(labels,probability))

def group_errors(labels, predictions):
    tn,fp,fn,tp=confusion_matrix(labels,predictions,labels=[0,1]).ravel()
    return dict(customers=len(labels),positives=int(tp+fn),negatives=int(tn+fp),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        false_negative_rate=float(fn/(fn+tp)) if fn+tp else None,
        false_positive_rate=float(fp/(fp+tn)) if fp+tn else None,
        error_rate=float((fp+fn)/len(labels)))

def main():
    root=Path(__file__).resolve().parents[1]; output=root/"outputs"
    raw=(output/"frozen_configuration.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FROZEN_SHA256:
        raise ValueError("Frozen configuration changed; stop before test evaluation")
    frozen=json.loads(raw)
    if frozen["csv_sha256"]!=CSV_SHA256 or frozen["scikit_learn_version"]!=sklearn.__version__:
        raise ValueError("Frozen dataset/software does not match")
    train,test,y_train,y_test=split_data(load_data())
    digest=hashlib.sha256(json.dumps(list(map(int,train.index))).encode()).hexdigest()
    assert digest==frozen["training_index_sha256"] and not set(train.index)&set(test.index)
    tables=output/"tables"; figures=output/"figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    configs=frozen["configurations"]+[dict(id="baseline",model="baseline")]
    scores=[]; errors=[]; groups=[]; selected=None
    roc_fig,roc_ax=plt.subplots(figsize=(9,7)); pr_fig,pr_ax=plt.subplots(figsize=(9,7))
    cm_fig,cm_axes=plt.subplots(2,4,figsize=(18,8))
    for config,ax in zip(configs,cm_axes.flat):
        pipeline=build_baseline() if config["model"]=="baseline" else build_default_models()[config["model"]]
        if config["model"]!="baseline": pipeline["model"].set_params(**config["parameters"])
        pipeline.fit(train,y_train)
        prediction=pipeline.predict(test); probability=pipeline.predict_proba(test)[:,1]
        assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
        train_prediction=pipeline.predict(train); train_probability=pipeline.predict_proba(train)[:,1]
        for partition,labels,pred,prob in [("train",y_train,train_prediction,train_probability),("test",y_test,prediction,probability)]:
            scores.append(dict(configuration=config["id"],partition=partition,**metrics(labels,pred,prob)))
        errors.append(dict(configuration=config["id"],**group_errors(y_test,prediction)))
        ConfusionMatrixDisplay.from_predictions(y_test,prediction,labels=[0,1],display_labels=["No","Yes"],ax=ax,colorbar=False)
        ax.set_title(config["id"].replace("_"," "),fontsize=10)
        RocCurveDisplay.from_predictions(y_test,probability,name=config["id"],ax=roc_ax)
        PrecisionRecallDisplay.from_predictions(y_test,probability,name=config["id"],ax=pr_ax)
        if config["id"]==frozen["selected_id"]:
            selected=pipeline
            for attribute in ["gender","SeniorCitizen"]:
                for value in sorted(test[attribute].unique()):
                    mask=test[attribute].eq(value).to_numpy()
                    groups.append(dict(attribute=attribute,group=str(value),**group_errors(y_test.to_numpy()[mask],prediction[mask])))
    cm_axes.flat[-1].axis("off")
    for fig,ax,name,title in [(roc_fig,roc_ax,"final_roc.png","Frozen configurations: held-out ROC"),(pr_fig,pr_ax,"final_precision_recall.png","Frozen configurations: held-out precision-recall")]:
        ax.set_title(title);fig.tight_layout();fig.savefig(figures/name,dpi=150);plt.close(fig)
    cm_fig.tight_layout();cm_fig.savefig(figures/"final_confusion_matrices.png",dpi=150);plt.close(cm_fig)
    pd.DataFrame(scores).to_csv(tables/"final_metrics.csv",index=False)
    pd.DataFrame(errors).to_csv(tables/"final_errors.csv",index=False)
    pd.DataFrame(groups).to_csv(tables/"selected_subgroup_errors.csv",index=False)
    # An explicitly synthetic example; never publish a real customer record.
    example=train.iloc[[0]].copy()
    for column in example: example[column]=train[column].mode().iloc[0]
    example["tenure"]=12.;example["MonthlyCharges"]=65.;example["TotalCharges"]=780.
    demonstration=dict(kind="synthetic; categorical training modes and specified numeric values",
        inputs=example.iloc[0].to_dict(),churn_probability=float(selected.predict_proba(example)[0,1]),
        prediction=int(selected.predict(example)[0]),selected_id=frozen["selected_id"])
    (output/"prediction_example.json").write_text(json.dumps(demonstration,indent=2)+"\n")
    print(pd.DataFrame(scores).query("partition == 'test'").to_string(index=False))

if __name__=="__main__": main()
