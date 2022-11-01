from ei_tensorflow import training
import cnn2snn

model = training.load_best_model(BEST_MODEL_PATH, akida_model=True)

#! Remove any dropout from the model, since it makes quantization-aware training hard
model = training.remove_dropout(model)

print('Performing post-training quantization...')

akida_model = cnn2snn.quantize(model,
                           weight_quantization=4,
                           activ_quantization=4,
                           input_weight_quantization=8)
print('Performing post-training quantization OK')
print('')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  mode='max',
                                                  verbose=1,
                                                  min_delta=0,
                                                  patience=10,
                                                  restore_best_weights=True)

print('Fine-tuning to recover accuracy...')
akida_model.compile(optimizer=opt,
                loss=fine_tune_loss,
                metrics=fine_tune_metrics)

akida_model.fit(train_dataset,
                epochs=30,
                verbose=2,
                validation_data=validation_dataset,
                callbacks=[early_stopping]
            )

print('Fine-tuning to recover accuracy OK')
print('')
