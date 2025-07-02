import numpy as np
import pandas as pd
from Rule_Validation import validate_rules
from Performance_measures import confusion_metrics_tf, Micro_calculate_measures, Macro_calculate_measures_tf

# Define validation parameters
n_iterations_validation = len(df_val) // batch_size
checkpoint_path = "./Capsnet_FFNN"

# Prepare validation data
X_val = df_val.iloc[:, :-1].to_numpy()
y_val_one = np.zeros((len(df_val), N_CLASSSES))
y_val_labels = df_val.iloc[:, -1].astype(int).to_numpy()
y_val_one[np.arange(len(y_val_labels)), y_val_labels] = 1

# Load extracted rules
with open("Experiments/extracted_rules.txt", "r") as f:
    rule_set = [eval(line.strip()) for line in f]
print("Number of rules loaded:", len(rule_set))

with tf.Session() as sess:
    # Restore the trained model
    saver.restore(sess, checkpoint_path)
    print("Model restored from checkpoint.")

    # Validation loop
    loss_vals = []
    acc_vals = []
    conf_vals = np.zeros((N_CLASSSES, N_CLASSSES))
    tp_v, tn_v, fp_v, fn_v = [], [], [], []
    pr_v, rc_v, f1_v = [], [], []
    y_true_val, y_pre_val = [], []

    btch_indx = 0
    for iteration in range(1, n_iterations_validation + 1):
        X_batch, y_batch = next_batch(df_val, btch_indx, batch_size)
        
        # Compute loss and accuracy
        loss_val, acc_val = sess.run(
            [cost, accuracy],
            feed_dict={X: X_batch, y: y_batch}
        )
        loss_vals.append(loss_val)
        acc_vals.append(acc_val)
        
        # Compute confusion metrics and macro measures
        conf, tp, tn, fp, fn, act, prob = confusion_metrics_tf(y, prediction, sess, {X: X_batch, y: y_batch})
        conf_vals = np.add(conf_vals, conf)
        tp_v.append(tp)
        tn_v.append(tn)
        fp_v.append(fp)
        fn_v.append(fn)
        y_true_val.extend(act)
        y_pre_val.extend(prob)
        
        pr, rc, f1 = Macro_calculate_measures_tf(y, prediction, sess, {X: X_batch, y: y_batch})
        pr_v.append(pr)
        rc_v.append(rc)
        f1_v.append(f1)
        
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            iteration, n_iterations_validation,
            iteration * 100 / n_iterations_validation),
            end=" " * 10)
        
        btch_indx += batch_size

    # Compute average metrics
    loss_val = np.mean(loss_vals)
    acc_val = np.mean(acc_vals)
    conf_val = conf_vals / n_iterations_validation
    tp_val = np.mean(tp_v)
    tn_val = np.mean(tn_v)
    fp_val = np.mean(fp_v)
    fn_val = np.mean(fn_v)
    pr_val = np.mean(pr_v)
    rc_val = np.mean(rc_v)
    f1_val = np.mean(f1_v)

    # Compute normalized confusion matrix
    sum_conf_val = np.sum(conf_val, axis=1)
    lst_val = []
    for i in range(len(sum_conf_val)):
        lst_val.append(np.round((conf_val[i, :] / sum_conf_val[i]), 2))
    arr_val = np.array(lst_val)

    # Compute micro measures
    val_measures = Micro_calculate_measures(tp_val, tn_val, fp_val, fn_val, 0)

    # Save results
    output = np.vstack((conf_val, arr_val))
    np.savetxt("Experiments/Micro_Validation_Conf_FFCN.csv", output, delimiter=',', fmt='%s')
    np.savetxt("Experiments/Micro_Validation_Measures_FFCN.csv", val_measures.to_numpy(), delimiter=',', fmt='%s')
    with open("Experiments/Macro_Validation_Results_FFCN.txt", 'w') as f:
        f_str = "PR:" + str(pr_val) + " RC:" + str(rc_val) + " F1:" + str(f1_val)
        f.write(f_str)
    np.savetxt("Experiments/y_true_FFCN_val.csv", np.array(y_true_val).flatten(), delimiter=',', fmt='%d')
    np.savetxt("Experiments/y_pre_FFCN_val.csv", np.array(y_pre_val).flatten(), delimiter=',' , fmt='%d')

    print("\nValidation completed!")
    print(f"Validation accuracy: {acc_val * 100:.4f}% ")
    print(f"Macro Precision: {pr_val:.4f}, Recall: {rc_val:.4f}, F1: {f1_val:.4f}")

    # Validate rules
    print("\nValidating rules on validation set...")
    validated_rules = validate_rules(rule_set, X_val, y_val_one)
    print("Number of rules after validation:", len(validated_rules))
