n_iterations_per_epoch = len(train_data) // batch_size
best_loss_val = np.inf
checkpoint_path = "./Capsnet_FFNN"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    
    epoch = 1
    loss_epochs = []
    rule_set = []
    
    while epoch < n_epochs:
        print(f"Epoch {epoch} started...")
        btch_indx = 0
        total_loss = 0
        
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = next_batch(df_train, btch_indx, batch_size)
            _, loss_train = sess.run([optimizer, cost], feed_dict={X: X_batch, y: y_batch})
            total_loss += loss_train

            '''
            if iteration % 10 == 0 or iteration == n_iterations_per_epoch:
                print(f"Iteration {iteration}/{n_iterations_per_epoch} ")
            '''

            print("\rIteration: {}/{} ({:.1f}%) ".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch
                ),
                end="")
            
            if epoch == (n_epochs - 1):
                pred_vect_lst = [pred.eval({X: X_batch, y: y_batch}) for pred in pred_lst]
                cpl_coeff_lst = [cpl.eval({X: X_batch, y: y_batch}) for cpl in cpl_lst]
                out_vect_lst = [out.eval({X: X_batch, y: y_batch}) for out in out_vect]
                pred = prediction.eval({X: X_batch, y: y_batch})
                rule_set.append(extract_rules_boundary(dt.eval({X: X_batch, y: y_batch}), cpl_coeff_lst, pred_vect_lst, out_vect_lst, pred))
            
            btch_indx += batch_size
        
        avg_loss = total_loss / n_iterations_per_epoch
        loss_epochs.append(avg_loss)
        print(f"Epoch {epoch} completed. \n")
        
        if avg_loss < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = avg_loss
            print("Model saved!\n")
        
        epoch += 1
    
    print("Training completed!")
    print("Extracted rules count:", len(rule_set))
    
    with open("Experiments/extracted_rules.txt", "w") as f:
        for rule in rule_set:
            f.write(str(rule) + "\n")
    print("Extracted rules saved to Experiments/extracted_rules.txt")
