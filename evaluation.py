'''
Evaluation

'''
# Example placeholder – adjust to how you're actually loading/generating rules

from Rule_Evaluation import evaluate_rules_boundary
from Performance_measures import confusion_metrics_tf, Micro_calculate_measures, Macro_calculate_measures_basic, Macro_calculate_measures_tf

n_iterations_test = len(test_data) // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    conf_tests = np.zeros((N_CLASSSES, N_CLASSSES))
    tp_t = []
    tn_t = []
    fp_t = []
    fn_t = []
    pr_t = []
    rc_t = []
    f1_t = []
    y_true_t = list()
    y_pre_t = list()
    btch_indx = 0
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = next_batch(df_test, btch_indx,batch_size)
        loss_test, acc_test = sess.run(
            [cost, accuracy],
            feed_dict={X: X_batch,
                       y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            iteration, n_iterations_test,
           iteration * 100 / n_iterations_test),
            end=" " * 10)
        conf, tp, tn, fp, fn, act, prob = confusion_metrics_tf(y, prediction, sess, {X: X_batch, y: y_batch})
        conf_tests = np.add(conf_tests, conf)
        tp_t.append(tp)
        tn_t.append(tn)
        fp_t.append(fp)
        fn_t.append(fn)
        y_true_t.extend(act)
        y_pre_t.extend(prob)
        pr, rc, f1 = Macro_calculate_measures_tf(y, prediction, sess, {X: X_batch, y: y_batch})
        pr_t.append(pr)
        rc_t.append(rc)
        f1_t.append(f1)
        btch_indx += batch_size
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)

    print("\rFinal test accuracy: {:.4f}%  ".format(
        acc_test * 100, loss_test))
    #######################Confusion Matrix & Loss##########################
    # confusion matrix: column: prediction labels and rows real labels
    conf_test = conf_tests / n_iterations_test
    tp_test = np.mean(tp_t)
    tn_test = np.mean(tn_t)
    fp_test = np.mean(fp_t)
    fn_test = np.mean(fn_t)
    pr_test = np.mean(pr_t)
    rc_test = np.mean(rc_t)
    f1_test = np.mean(f1_t)
    sum_conf_test = np.sum(conf_test, axis=1)
    lst_test = []
    for i in range(len(sum_conf_test)):
        lst_test.append(np.round((conf_test[i, :] / sum_conf_test[i]), 2))
    arr_test = np.array(lst_test)
    test_measures = Micro_calculate_measures(tp_test, tn_test, fp_test, fn_test, 0)
    output = np.vstack((conf_test, arr_test))
    np.savetxt("Experiments/Micro_Test_Conf_FFCN.csv", output, delimiter=',', fmt='%s')
    np.savetxt("Experiments/Micro_Test_Measures_FFCN.csv", test_measures.to_numpy(), delimiter=',', fmt='%s')
    f = open("Experiments/Macro_Test_Results_FFCN.txt", 'w')
    f_str = "PR:" + str(pr_test) + " RC:" + str(rc_test) + " F1:" + str(f1_test)
    f.write(f_str)
    f.close()
    np.savetxt("Experiments/y_true_FFCN_test.csv", np.array(y_true_t).flatten(), delimiter=',', fmt='%d')
    np.savetxt("Experiments/y_pre_FFCN_test.csv", np.array(y_pre_t).flatten(), delimiter=',', fmt='%d')

     #########################################################################
     # Rule Evaluation
    test_data = df_test.iloc[:, :-1].to_numpy()
    test_labels = df_test.iloc[:, -1].to_numpy()
    
    # Assuming test_labels contains integer labels (0, 1, ..., N-1)
    if not np.issubdtype(test_labels.dtype, np.integer):
        test_labels = test_labels.astype(int)
    y_test_one = np.zeros((len(test_labels), N_CLASSSES))
    y_test_one[np.arange(len(test_labels)), test_labels] = 1
    
    # Then call the function
    evaluate_rules_boundary(validated_rules, test_data, y_test_one)

   
