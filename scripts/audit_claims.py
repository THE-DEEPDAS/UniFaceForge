"""Derive the paper's reported metrics from its stated counts.

This is intentionally independent of the neural network: it detects whether
the manuscript's numbers are mutually consistent before any training run.
"""
import json

def metrics(tn, fp, fn, tp):
    n=tn+fp+fn+tp
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*precision*recall/(precision+recall)
    return {"n":n,"accuracy":(tn+tp)/n,"precision":precision,"recall":recall,"f1":f1,
            "false_positive_rate":fp/(fp+tn),"false_negative_rate":fn/(fn+tp)}

def main():
    cases={"figure_5_494_495":metrics(495,5,6,494),
           "alternative_496_495":metrics(496,4,5,495)}
    dataset={"real":100000,"traditional":79862,"gan":92247,"synthetic":15076}
    dataset["fake_total"]=sum(dataset[k] for k in ("traditional","gan","synthetic"))
    dataset["total"]=dataset["real"]+dataset["fake_total"]
    dataset["15_percent_test"]=round(dataset["total"]*.15)
    print(json.dumps({"confusion_matrix_derivations":cases,"dataset_counts":dataset,
                      "reported_claims":{"precision":.991,"recall":.989,"f1":.990,
                                         "auc":.998,"eer":.0082}},indent=2))

if __name__ == "__main__": main()
